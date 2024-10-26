# %%
import numpy as np
import torch
import math
from tqdm import tqdm
from torch.autograd import grad

class HMC_U1:
    def __init__(self, lattice_size, beta, n_thermalization_steps, n_steps, step_size, device='cpu'):
        """
        Initialize the HMC_U1 class.
        
        Parameters:
        -----------
        lattice_size : int
            The size of the lattice (assumed to be square).
        beta : float
            The inverse coupling constant.
        n_thermalization_steps : int
            The number of thermalization steps.
        n_steps : int
            The number of leapfrog steps in each HMC trajectory.
        step_size : float
            The step size for each leapfrog step.
        device : str
            The device to use for computation ('cpu' or 'cuda').
        """
        self.lattice_size = lattice_size
        self.beta = beta
        self.n_thermalization_steps = n_thermalization_steps
        self.n_steps = n_steps
        self.dt = step_size
        self.device = device
        self.lat = [lattice_size, lattice_size]
        self.nd = len(self.lat)

        # Set default data type and device
        torch.set_default_dtype(torch.float64)  # Equivalent to torch.DoubleTensor
        torch.set_default_device(self.device)   # Set the default device
        torch.manual_seed(1331)

    def initialize(self):
        """
        Initialize the field configuration (theta) uniformly between -pi and pi.
        """
        return torch.empty([self.nd] + self.lat, device=self.device).uniform_(-math.pi, math.pi)

    def plaqphase(self, theta):
        """
        Compute plaquette phase: 
        P = U0(x,y) * U1(x+1,y) * Udagger0(x,y+1) * Udagger1(x,y)
        It corresponds to the calculation of angle as:
        theta_P = theta0(x,y) + theta1(x+1,y) - theta0(x,y+1) - theta1(x,y)
        """
        theta0, theta1 = theta[0], theta[1]
        return (theta0 - theta1 - torch.roll(theta0, shifts=-1, dims=1) + torch.roll(theta1, shifts=-1, dims=0))

    def action(self, theta):
        """
        Compute the action (negative log likelihood) of the field configuration.
        
        The action is given by:
        Re[1 - exp(i * theta_P)] = 1 - cos(theta_P)
        We drop the constant term to get the action as:
        -beta * sum(cos(theta_P))
        """
        theta_P = self.plaqphase(theta)
        return (-self.beta) * torch.sum(torch.cos(theta_P))

    def force(self, theta):
        """
        Compute the force (gradient of the action) for the current field configuration.
        """
        theta.requires_grad_(True)
        s = self.action(theta)
        force = grad(s, theta, create_graph=False, retain_graph=False)[0]
        theta.requires_grad_(False)
        return force

    def leapfrog(self, theta, pi):
        """
        Perform leapfrog integration to numerically integrate the Hamiltonian equations for coordinates and momenta.

        The leapfrog integration is used to propose new field configurations by simulating the dynamics of the system under a fictitious Hamiltonian. This method is symplectic and time-reversible, which are important properties for maintaining detailed balance in the HMC algorithm.

        Parameters:
        theta (numpy.ndarray): Current field configuration (coordinates).
        pi (numpy.ndarray): Current conjugate momenta.

        Returns:
        tuple: A pair (theta_new, pi_new) containing the updated field configuration and conjugate momenta.
        """
        pi = pi - 0.5 * self.dt * self.force(theta)
        for _ in range(self.n_steps):
            theta = theta + self.dt * pi
            theta = self.regularize(theta)
            force = self.force(theta)
            pi = pi - self.dt * force
        theta = theta + 0.5 * self.dt * pi
        return theta, pi

    def metropolis_step(self, theta_old):
        # Generate random momenta from a Gaussian distribution
        pi = torch.randn_like(theta_old, device=self.device)
        
        # Calculate the initial Hamiltonian
        H_old = self.action(theta_old) + 0.5 * torch.sum(pi ** 2)
        
        # Perform leapfrog integration to get new theta and pi
        theta_new, pi_new = self.leapfrog(theta_old.clone(), pi.clone())
        
        # Calculate the new Hamiltonian
        H_new = self.action(theta_new) + 0.5 * torch.sum(pi_new ** 2)
        
        # Calculate the change in Hamiltonian
        delta_H = H_new - H_old
        
        # Calculate the acceptance probability
        accept_prob = torch.exp(-delta_H)
        
        # Metropolis acceptance step
        if torch.rand([], device=self.device) < accept_prob:
            # Accept the new configuration
            return theta_new, True, H_new.item()
        else:
            # Reject and keep the old configuration
            return theta_old, False, H_old.item()

    def regularize(self, theta):
        """
        Regularize the angle to be within the range [-pi, pi].
        """
        return theta - 2 * math.pi * torch.floor((theta + math.pi) / (2 * math.pi))

    def topological_charge(self, theta):
        """
        Compute the topological charge of the field configuration.
        
        In the continuous theory, the topological charge is given by Q = 1/(2π) * \int d^2x F_{01}(x) = 1/(2π) * \int d^2x \partial_0 A_1(x) - \partial_1 A_0(x).

        In the discrete theory, the topological charge is given by Q = 1/(2π) * sum(theta_P), where theta_P is the angle deficit of the plaquette P.
        """
        theta_P = self.plaqphase(theta)
        theta_P_wrapped = self.regularize(theta_P)
        Q = torch.floor(0.1 + torch.sum(theta_P_wrapped) / (2 * math.pi))
        return Q.item()

    def thermalize(self, field_transformation=None):
        theta = self.initialize()
        actions = []
        acceptance_count = 0

        for _ in tqdm(range(self.n_thermalization_steps), desc="Thermalizing"):
            if field_transformation:
                theta = field_transformation(theta)
            theta, accepted, _ = self.metropolis_step(theta)
            actions.append(self.action(theta).item())
            if accepted:
                acceptance_count += 1

        acceptance_rate = acceptance_count / self.n_thermalization_steps
        return theta, torch.tensor(actions), acceptance_rate

    def run(self, n_iterations, theta_thermalized):
        theta = theta_thermalized
        actions = []
        hamiltonians = []
        acceptance_count = 0
        topological_charges = []

        for _ in tqdm(range(n_iterations), desc="Running HMC"):
            theta, accepted, H = self.metropolis_step(theta)
            actions.append(self.action(theta).item())
            hamiltonians.append(H)
            topological_charges.append(self.topological_charge(theta))
            if accepted:
                acceptance_count += 1

        acceptance_rate = acceptance_count / n_iterations
        return (theta, torch.tensor(actions), acceptance_rate, 
                torch.tensor(topological_charges), torch.tensor(hamiltonians))

if __name__ == "__main__":
    from utils import plot_results, compute_autocorrelation, compute_autocorrelation_by_def, plaquette_value, calculate_plaquette_from_field
    import numpy as np
    
    # Set simulation parameters
    lattice_size = 16
    volume = lattice_size ** 2
    beta = 3
    n_thermalization_steps = 100
    n_steps = 100
    step_size = 0.01
    n_iterations = 200

    # Initialize HMC
    hmc = HMC_U1(lattice_size, beta, n_thermalization_steps, n_steps, step_size)

    print(">>> Starting thermalization")
    theta_thermalized, thermalization_actions, thermalization_acceptance_rate = hmc.thermalize()

    print(">>> Running HMC")
    final_config, actions, acceptance_rate, topological_charges, hamiltonians = hmc.run(n_iterations, theta_thermalized)

    # Convert to numpy arrays
    thermalization_actions = thermalization_actions.cpu().numpy()
    actions = actions.cpu().numpy()
    hamiltonians = hamiltonians.cpu().numpy()
    final_config = final_config.cpu().numpy()
    topological_charges = topological_charges.cpu().numpy()

    # Calculate expected and actual plaquette values
    expected_plaquette = plaquette_value(beta)
    real_plaquette = calculate_plaquette_from_field(final_config)
    print(f"Expected plaquette value for beta = {beta}: {expected_plaquette}")
    print(f"Real plaquette value from final configuration: {real_plaquette}")

    # Compute autocorrelation of topological charges
    max_lag = 20
    autocorrelations = compute_autocorrelation(topological_charges, max_lag, beta, volume)
    autocorrelations_by_def = compute_autocorrelation_by_def(topological_charges, max_lag)

    # Plot results
    plot_results(thermalization_actions, actions, topological_charges, hamiltonians, autocorrelations, title_suffix="(Using Infinite Volume Susceptibility)")
    plot_results(thermalization_actions, actions, topological_charges, hamiltonians, autocorrelations_by_def, title_suffix="(Using Autocorrelation by Definition)")

    # Print acceptance rates
    print(f"Thermalization acceptance rate: {thermalization_acceptance_rate:.4f}")
    print(f"Acceptance rate: {acceptance_rate:.4f}")

    # Calculate and print relative change in Hamiltonian (delta H)
    delta_H = np.max(hamiltonians) - np.min(hamiltonians)
    H_mean = np.mean(hamiltonians)
    relative_delta_H = delta_H / H_mean
    print(f"Relative variation of Hamiltonian (delta H / H_mean): {relative_delta_H:.4f}")

    # Calculate and print change in topological charge (delta Q)
    delta_Q = np.max(topological_charges) - np.min(topological_charges)
    print(f"Variation of topological charge (delta Q): {delta_Q:.4f}")

    print(">>> Simulation completed")
# %%
