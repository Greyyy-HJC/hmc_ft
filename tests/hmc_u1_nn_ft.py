# %%
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import plaq_from_field, topo_from_field, plaq_mean_from_field
from nn_model import NNFieldTransformation

class HMC_U1_NN_FT:
    def __init__(
        self,
        lattice_size,
        beta,
        n_thermalization_steps,
        n_steps,
        step_size,
        nn_transformation,
        device="cpu",
    ):
        """
        Initialize the HMC with neural network field transformation.

        Parameters:
        -----------
        lattice_size : int
            The size of the lattice (assumed to be square)
        beta : float
            The inverse coupling constant
        n_thermalization_steps : int
            The number of thermalization steps
        n_steps : int
            The number of leapfrog steps in each HMC trajectory
        step_size : float
            The step size for each leapfrog step
        nn_transformation : NNFieldTransformation
            The trained neural network field transformation
        device : str
            The device to use for computation ('cpu' or 'cuda')
        """
        self.lattice_size = lattice_size
        self.beta = beta
        self.n_thermalization_steps = n_thermalization_steps
        self.n_steps = n_steps
        self.dt = step_size
        self.device = torch.device(device)
        
        # Get the transformation functions from trained NN
        self.field_transform = nn_transformation.get_field_transformation()
        self.transformed_action = nn_transformation.get_transformed_action(beta)

        # Set default data type and device
        torch.set_default_dtype(torch.float32)
        torch.set_default_device(self.device)
        torch.manual_seed(1331)

    def initialize(self):
        """Initialize the field configuration"""
        return torch.zeros([2, self.lattice_size, self.lattice_size])

    def force(self, theta):
        """
        Compute the force (gradient of transformed action) using autograd.
        """
        theta.requires_grad_(True)
        action = self.transformed_action(theta)
        force = torch.autograd.grad(action, theta)[0]
        theta.requires_grad_(False)
        return force

    def leapfrog(self, theta, pi):
        """
        Perform leapfrog integration steps.
        """
        dt = self.dt
        theta_ = theta + 0.5 * dt * pi
        pi_ = pi - dt * self.force(theta_)
        for _ in range(self.n_steps - 1):
            theta_ = theta_ + dt * pi_
            pi_ = pi_ - dt * self.force(theta_)
        theta_ = theta_ + 0.5 * dt * pi_
        return theta_, pi_

    def metropolis_step(self, theta):
        """
        Perform one Metropolis step using HMC.
        """
        pi = torch.randn_like(theta, device=self.device)
        
        # Compute initial Hamiltonian
        action_initial = self.transformed_action(theta)
        H_old = action_initial + 0.5 * torch.sum(pi**2)

        # Perform leapfrog integration
        new_theta, new_pi = self.leapfrog(theta.clone(), pi.clone())
        
        # Compute final Hamiltonian
        action_final = self.transformed_action(new_theta)
        H_new = action_final + 0.5 * torch.sum(new_pi**2)

        # Metropolis acceptance step
        delta_H = H_new - H_old
        accept_prob = torch.exp(-delta_H)
        
        if torch.rand([], device=self.device) < accept_prob:
            return new_theta, True, H_new.item()
        else:
            return theta, False, H_old.item()

    def thermalize(self):
        """
        Perform thermalization steps.
        """
        theta = self.initialize()
        plaq_ls = []
        acceptance_count = 0

        for _ in tqdm(range(self.n_thermalization_steps), desc="Thermalizing"):
            # Transform theta to original space for plaquette calculation
            theta_old = self.field_transform(theta)
            plaq = plaq_mean_from_field(theta_old).item()
            
            theta, accepted, _ = self.metropolis_step(theta)
            
            plaq_ls.append(plaq)
            if accepted:
                acceptance_count += 1

        acceptance_rate = acceptance_count / self.n_thermalization_steps
        return theta, plaq_ls, acceptance_rate

    def run(self, n_iterations, theta, store_interval=1):
        """
        Run HMC simulation.

        Parameters:
        -----------
        n_iterations : int
            Number of HMC iterations to run
        theta : tensor
            Initial field configuration
        store_interval : int
            Store results every store_interval iterations
        """
        plaq_ls = []
        hamiltonians = []
        acceptance_count = 0
        topological_charges = []

        for i in tqdm(range(n_iterations), desc="Running HMC"):
            theta, accepted, H = self.metropolis_step(theta)
            
            if i % store_interval == 0:
                # Transform theta to original space for measurements
                theta_old = self.field_transform(theta)
                plaq = plaq_mean_from_field(theta_old).item()
                plaq_ls.append(plaq)
                hamiltonians.append(H)
                topological_charges.append(topo_from_field(theta_old).item())
                
            if accepted:
                acceptance_count += 1

        acceptance_rate = acceptance_count / n_iterations
        return (
            theta,
            plaq_ls,
            acceptance_rate,
            topological_charges,
            hamiltonians,
        ) 

def main():
    # HMC parameters
    lattice_size = 8
    beta = 4.0
    n_thermalization = 20
    n_trajectories = 500
    n_steps = 20
    step_size = 0.2
    store_interval = 10

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize and train the neural network transformation
    print("\nInitializing and training neural network field transformation...")
    nn_transformation = NNFieldTransformation(
        lattice_size=lattice_size,
        model_type='CNN',
        epsilon=0.01,
        epsilon_decay=1.0,
        device=device
    )
    nn_transformation.train(beta=beta, n_iterations=200)

    # Initialize HMC with the trained transformation
    print("\nInitializing HMC with neural network field transformation...")
    hmc = HMC_U1_NN_FT(
        lattice_size=lattice_size,
        beta=beta,
        n_thermalization_steps=n_thermalization,
        n_steps=n_steps,
        step_size=step_size,
        nn_transformation=nn_transformation,
        device=device
    )

    # Thermalization
    print("\nStarting thermalization...")
    theta, plaq_therm, acceptance_rate_therm = hmc.thermalize()
    print(f"Thermalization acceptance rate: {acceptance_rate_therm:.3f}")

    # Production run
    print("\nStarting production run...")
    theta, plaq_ls, acceptance_rate, topo_charges, hamiltonians = hmc.run(
        n_iterations=n_trajectories,
        theta=theta,
        store_interval=store_interval
    )
    print(f"Production acceptance rate: {acceptance_rate:.3f}")

    # Plot results
    import matplotlib.pyplot as plt

    # Plot plaquette history
    plt.figure(figsize=(10, 6))
    plt.plot(plaq_ls)
    plt.xlabel('Configuration (every 10 trajectories)')
    plt.ylabel('Plaquette')
    plt.title('Plaquette History')
    plt.show()

    # Plot topological charge history
    plt.figure(figsize=(10, 6))
    plt.plot(topo_charges)
    plt.xlabel('Configuration (every 10 trajectories)')
    plt.ylabel('Topological Charge')
    plt.title('Topological Charge History')
    plt.show()

    # Plot Hamiltonian history
    plt.figure(figsize=(10, 6))
    plt.plot(hamiltonians)
    plt.xlabel('Configuration (every 10 trajectories)')
    plt.ylabel('Hamiltonian')
    plt.title('Hamiltonian History')
    plt.show()

    # Print statistics
    print("\nSimulation Statistics:")
    print(f"Mean plaquette: {torch.tensor(plaq_ls).mean():.6f} ± {torch.tensor(plaq_ls).std():.6f}")
    print(f"Mean topological charge: {torch.tensor(topo_charges).mean():.6f} ± {torch.tensor(topo_charges).std():.6f}")
    print(f"Mean Hamiltonian: {torch.tensor(hamiltonians).mean():.6f} ± {torch.tensor(hamiltonians).std():.6f}")

if __name__ == "__main__":
    main()
# %%
