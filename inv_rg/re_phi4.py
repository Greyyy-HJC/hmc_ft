# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

# Parameters
L32 = 32  # Original lattice size
L16 = 16  # Downscaled lattice size
num_configs = 1000  # Number of configurations
thermalization_steps = 1000  # Steps for thermalization
kappa_L = 1.0
lambda_L = 0.7
mu2_L_central = -0.9515  # Central value for simulations
mu2_L_range = np.linspace(-0.96, -0.94, 20)  # Range for reweighting
n_steps = 20  # Number of leapfrog steps
epsilon = 0.0005  # Leapfrog step size

# Initialize lattice
def initialize_lattice(L):
    # Return as torch tensor
    return torch.rand((L, L), dtype=torch.float64) * 2 - 1

def compute_potential(phi, kappa_L, mu2_L, lambda_L):
    """Compute the potential energy of the configuration using PyTorch"""
    # Ensure phi is a torch tensor with gradients enabled
    if not isinstance(phi, torch.Tensor):
        phi = torch.tensor(phi, dtype=torch.float64, requires_grad=True)
    elif not phi.requires_grad:
        phi.requires_grad_(True)
    
    # Calculate neighbors using roll operations
    neighbors = (
        torch.roll(phi, 1, 0) + torch.roll(phi, -1, 0) +
        torch.roll(phi, 1, 1) + torch.roll(phi, -1, 1)
    )
    
    # Calculate potential terms
    kinetic_term = -kappa_L * phi * neighbors
    mass_term = (mu2_L + 4 * kappa_L) * phi**2 / 2
    interaction_term = lambda_L * phi**4 / 4
    
    # Sum all terms
    potential = torch.sum(kinetic_term + mass_term + interaction_term)
    return potential

def compute_force(phi, kappa_L, mu2_L, lambda_L):
    """Compute the force (-dV/dphi) using PyTorch autograd"""
    # Convert to tensor if needed
    if not isinstance(phi, torch.Tensor):
        phi = torch.tensor(phi, dtype=torch.float64, requires_grad=True)
    elif not phi.requires_grad:
        phi.requires_grad_(True)
    
    # Compute potential and its gradient
    potential = compute_potential(phi, kappa_L, mu2_L, lambda_L)
    force = -torch.autograd.grad(potential, phi)[0]  # Negative gradient gives the force
    
    return force.detach()  # Detach to remove gradient history

def leapfrog_step(phi, pi, epsilon, kappa_L, mu2_L, lambda_L):
    """Perform one leapfrog integration step using PyTorch"""
    # Half step in momentum
    force = compute_force(phi, kappa_L, mu2_L, lambda_L)
    pi -= epsilon * force / 2
    
    # Full step in position and momentum
    for _ in range(n_steps - 1):
        phi += epsilon * pi
        force = compute_force(phi, kappa_L, mu2_L, lambda_L)
        pi -= epsilon * force

    # Full step in position
    phi += epsilon * pi
    
    # Half step in momentum
    force = compute_force(phi, kappa_L, mu2_L, lambda_L)
    pi -= epsilon * force / 2
    
    return phi, pi

def hmc_update(phi, kappa_L, mu2_L, lambda_L):
    """Perform one HMC update step using PyTorch"""
    # L = phi.shape[0]
    
    # Initial momentum and energy
    pi = torch.randn_like(phi)
    H_old = torch.sum(pi**2) / 2 + compute_potential(phi, kappa_L, mu2_L, lambda_L)
    
    # Make copies for the trajectory
    phi_new = phi.clone()
    pi_new = pi.clone()
    
    # Perform leapfrog integration
    phi_new, pi_new = leapfrog_step(phi_new, pi_new, epsilon, kappa_L, mu2_L, lambda_L)
    
    # Compute new energy
    H_new = torch.sum(pi_new**2) / 2 + compute_potential(phi_new, kappa_L, mu2_L, lambda_L)
    
    # Metropolis accept/reject
    dH = H_new - H_old
    if dH < 0 or torch.rand(1) < torch.exp(-dH):
        return phi_new, 1
    return phi, 0

def generate_configs(L, num_configs, kappa_L, mu2_L, lambda_L, thermalization_steps, step_gap):
    """Generate Monte Carlo configurations with HMC updates using PyTorch"""
    phi = initialize_lattice(L)
    acceptance_rates = []
    configs = []

    # Thermalization
    for _ in tqdm.tqdm(range(thermalization_steps), desc="Thermalization"):
        phi, accepted = hmc_update(phi, kappa_L, mu2_L, lambda_L)
        acceptance_rates.append(accepted)
    print(f"Mean acceptance rate during thermalization: {np.mean(acceptance_rates):.2f}")

    # Generate configurations
    for _ in tqdm.tqdm(range(num_configs), desc="Generating configurations"):
        accepted_count = 0
        for _ in range(step_gap):
            phi, accepted = hmc_update(phi, kappa_L, mu2_L, lambda_L)
            accepted_count += accepted
        # Store configuration as numpy array
        configs.append(phi.detach().numpy())
        acceptance_rates.append(accepted_count / step_gap)

    print(f"Mean acceptance rate after thermalization: {np.mean(acceptance_rates):.2f}")
    return configs


# RG Transformation with sign determination
def rg_transform(configs):
    new_configs = []
    for config in configs:
        new_L = config.shape[0] // 2
        new_config = np.zeros((new_L, new_L))
        for i in range(new_L):
            for j in range(new_L):
                block = config[2*i:2*i+2, 2*j:2*j+2]
                block_mean = np.mean(block)
                block_sign = np.sign(block_mean)
                block_magnitude = np.mean(block[block > 0]) if block_sign > 0 else np.mean(block[block < 0])
                new_config[i, j] = block_magnitude
        new_configs.append(new_config)
    return new_configs

# Histogram reweighting (Eq. 7)
def histogram_reweighting(configs, mu2_L_old, mu2_L_new):
    """
    Implement histogram reweighting according to Eq. 7 in the paper.
    Here, we're reweighting with respect to mu2_L (mass parameter),
    so K_m - K_m^(0) is (mu2_L_new - mu2_L_old) and S^(m) is the mass term phi^2/2
    """
    weights = []
    observables = []
    
    for config in configs:
        # Convert to torch tensor if needed
        if not isinstance(config, torch.Tensor):
            config = torch.tensor(config, dtype=torch.float64)
            
        # Compute the mass term S^(m) = phi^2/2
        S_m = torch.sum(config**2) / 2
        
        # Compute the reweighting factor exp[-(K_m - K_m^(0))S^(m)]
        dK = mu2_L_new - mu2_L_old
        weight = torch.exp(-dK * S_m)
        
        # Store weight and observable (magnetization in this case)
        weights.append(weight.item())
        observables.append(torch.abs(torch.mean(config)).item())
    
    weights = np.array(weights)
    observables = np.array(observables)
    
    # Compute reweighted expectation value according to Eq. 7
    reweighted_value = np.sum(weights * observables) / np.sum(weights)
    
    return reweighted_value

# Main workflow
configs_L32 = generate_configs(L32, num_configs, kappa_L, mu2_L_central, lambda_L, thermalization_steps, step_gap=10)
print(f"Generated {len(configs_L32)} L32 configs")
# Calculate magnetization distribution for L32 configs

configs_L16 = generate_configs(L16, num_configs, kappa_L, mu2_L_central, lambda_L, thermalization_steps, step_gap=10)
print(f"Generated {len(configs_L16)} L16 configs")

configs_RG = rg_transform(configs_L32)
print(f"RG transformed {len(configs_RG)} L32 configs")


# %%
magnetizations_L16 = [np.abs(np.mean(config)) for config in configs_L16]
magnetizations_L32 = [np.abs(np.mean(config)) for config in configs_L32]
magnetizations_RG = [np.abs(np.mean(config)) for config in configs_RG]


print(magnetizations_L32[:5])
print(magnetizations_L16[:5])
print(magnetizations_RG[:5])


print(configs_L16[0][:5, :5])



# %%
# Separate configs into bins
n_bins = 10
bin_size = num_configs // n_bins
configs_L16_binned = [configs_L16[i:i+bin_size] for i in range(0, num_configs, bin_size)]
configs_RG_binned = [configs_RG[i:i+bin_size] for i in range(0, num_configs, bin_size)]

# Compute reweighted observables for each bin
m_original_binned = []
m_rg_binned = []

for mu2_L in mu2_L_range:
    m_o_bins = []
    m_r_bins = []
    
    for bin_idx in range(n_bins):
        m_o = histogram_reweighting(configs_L16_binned[bin_idx], mu2_L_central, mu2_L)
        m_r = histogram_reweighting(configs_RG_binned[bin_idx], mu2_L_central, mu2_L)
        m_o_bins.append(m_o)
        m_r_bins.append(m_r)
    
    m_original_binned.append(m_o_bins)
    m_rg_binned.append(m_r_bins)

# Convert to numpy arrays
m_original_binned = np.array(m_original_binned)  # Shape: (n_mu2_points, n_bins)
m_rg_binned = np.array(m_rg_binned)  # Shape: (n_mu2_points, n_bins)

# Calculate means and standard errors
m_original = np.mean(m_original_binned, axis=1)
m_rg = np.mean(m_rg_binned, axis=1)
errors_original = np.std(m_original_binned, axis=1) / np.sqrt(n_bins)  # Standard error of the mean
errors_rg = np.std(m_rg_binned, axis=1) / np.sqrt(n_bins)  # Standard error of the mean

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(mu2_L_range, m_original, label=r"$L = 16$", color="blue")
plt.fill_between(mu2_L_range, m_original - errors_original, m_original + errors_original, color="blue", alpha=0.3)
plt.plot(mu2_L_range, m_rg, label=r"$L' = 16$", color="red", linestyle="--")
plt.fill_between(mu2_L_range, m_rg - errors_rg, m_rg + errors_rg, color="red", alpha=0.3)
plt.xlabel(r"Dimensionless squared mass $\mu_L^2$")
plt.ylabel(r"Magnetization $|m|$")
plt.title("Magnetization vs. Dimensionless Mass")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
