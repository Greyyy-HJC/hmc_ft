# %%
import torch
from tqdm import tqdm
from utils import plaq_from_field, topo_from_field, plaq_mean_from_field, regularize

def action(theta, beta):
    theta_P = plaq_from_field(theta)
    thetaP_wrapped = regularize(theta_P)
    action_value = (-beta) * torch.sum(torch.cos(thetaP_wrapped))
    
    assert action_value.dim() == 0, "Action value is not a scalar."
    
    return action_value


class HMC_U1:
    def __init__(
        self,
        lattice_size,
        beta,
        n_thermalization_steps,
        n_steps,
        step_size,
        device="cpu",
    ):
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
        n_threads : int
            Number of OpenMP threads to use
        n_interop_threads : int
            Number of interop threads to use
        """
        self.lattice_size = lattice_size
        self.beta = beta
        self.n_thermalization_steps = n_thermalization_steps
        self.n_steps = n_steps
        self.dt = step_size
        self.device = torch.device(device)

        # Set default data type and device
        torch.set_default_dtype(torch.float32)
        torch.set_default_device(self.device)
        torch.manual_seed(1331)

    def initialize(self):
        return torch.zeros([2, self.lattice_size, self.lattice_size])

    def force(self, theta):
        theta.requires_grad_(True)
        action_value = action(theta, self.beta)
        action_value.backward()
        ff = theta.grad
        theta.requires_grad_(False) # so that the memory can be freed
        return ff

    def leapfrog(self, theta, pi):
        dt = self.dt
        theta_ = theta + 0.5 * dt * pi
        pi_ = pi - dt * self.force(theta_)
        for _ in range(self.n_steps - 1):
            theta_ = theta_ + dt * pi_
            pi_ = pi_ - dt * self.force(theta_)
        theta_ = theta_ + 0.5 * dt * pi_
        theta_ = regularize(theta_)
        return theta_, pi_

    def metropolis_step(self, theta):
        pi = torch.randn_like(theta, device=self.device)
        action_value = action(theta, self.beta)
        H_old = action_value + 0.5 * torch.sum(pi**2)

        new_theta, new_pi = self.leapfrog(theta.clone(), pi.clone())
        new_action_value = action(new_theta, self.beta)
        H_new = new_action_value + 0.5 * torch.sum(new_pi**2)

        delta_H = H_new - H_old
        accept_prob = torch.exp(-delta_H)

        if torch.rand([], device=self.device) < accept_prob:
            return new_theta, True, H_new.item()
        else:
            return theta, False, H_old.item()

    def tune_step_size(self, n_tune_steps=1000, target_rate=0.65, target_tolerance=0.15, initial_step_size=0.2, max_attempts=10, theta=None):
        """
        Tune the step size to achieve desired acceptance rate using binary search.
        
        Parameters:
        -----------
        n_tune_steps : int
            Number of steps to use for tuning
        target_rate : float
            Target acceptance rate (default: 0.65)
        target_tolerance : float
            Acceptable deviation from target rate (default: 0.15)
        initial_step_size : float
            Initial step size to start tuning from
        max_attempts : int
            Maximum number of tuning attempts
        theta : tensor
            The theta to use for tuning (optional, defaults to initialized theta)
        """
        if theta is None:
            theta = self.initialize()
        else:
            theta = theta.clone()  # Don't modify the input theta
        
        self.dt = initial_step_size
        step_min = 1e-6
        step_max = 1.0
        best_dt = self.dt
        best_rate_diff = float('inf')
        
        for attempt in range(max_attempts):
            acceptance_count = 0
            for _ in tqdm(range(n_tune_steps), desc=f"Tuning step size (attempt {attempt+1}/{max_attempts})"):
                _, accepted, _ = self.metropolis_step(theta)
                if accepted:
                    acceptance_count += 1
            
            current_rate = acceptance_count / n_tune_steps
            rate_diff = abs(current_rate - target_rate)
            print(f"Step size: {self.dt:.6f}, Acceptance rate: {current_rate:.2%}")
            
            # Save best result so far
            if rate_diff < best_rate_diff:
                best_dt = self.dt
                best_rate_diff = rate_diff
            
            # Check if current rate is acceptable
            if abs(current_rate - target_rate) <= target_tolerance:
                print(f"Found good step size: {self.dt:.6f}")
                break
            
            # Binary search update
            if current_rate > target_rate:
                step_min = self.dt
                self.dt = min((self.dt + step_max) / 2, step_max)
            else:
                step_max = self.dt
                self.dt = max((self.dt + step_min) / 2, step_min)
        
        # Use best found step size if we didn't converge
        if abs(current_rate - target_rate) > target_tolerance:
            print(f"Using best found step size: {best_dt:.6f}")
            self.dt = best_dt

    def thermalize(self):
        """
        First do a rough thermalization, then tune step size, then do final thermalization.
        """
        # Initial thermalization to get away from cold start
        theta = self.initialize()
        n_initial_therm = self.n_thermalization_steps
        
        print(">>> Initial thermalization...")
        for _ in tqdm(range(n_initial_therm), desc="Initial thermalization"):
            theta, _, _ = self.metropolis_step(theta)
        
        # Tune step size on thermalized configuration
        print(">>> Tuning step size...")
        self.tune_step_size(theta=theta)  # Pass the thermalized theta
        
        # Final thermalization with tuned step size
        print(">>> Final thermalization...")
        plaq_ls = []
        acceptance_count = 0
        
        for _ in tqdm(range(self.n_thermalization_steps), desc="Final thermalization"):
            plaq = plaq_mean_from_field(theta).item()
            theta, accepted, _ = self.metropolis_step(theta)
            
            plaq_ls.append(plaq)
            if accepted:
                acceptance_count += 1

        acceptance_rate = acceptance_count / self.n_thermalization_steps
        return theta, plaq_ls, acceptance_rate

    def run(self, n_iterations, theta, store_interval=1):
        """
        Parameters:
        -----------
        n_iterations : int
            Number of HMC iterations to run
        theta : tensor
            Initial field configuration
        store_interval : int
            Store results every store_interval iterations to save memory
        """
        theta_ls = []
        plaq_ls = []
        hamiltonians = []
        acceptance_count = 0
        topological_charges = []

        for i in tqdm(range(n_iterations), desc="Running HMC"):
            theta, accepted, H_val = self.metropolis_step(theta)
            
            if i % store_interval == 0:  # only store data at specific intervals
                theta_ls.append(theta)
                plaq = plaq_mean_from_field(theta).item()
                plaq_ls.append(plaq)
                hamiltonians.append(H_val)
                topological_charges.append(topo_from_field(theta).item())

            if accepted:
                acceptance_count += 1

        acceptance_rate = acceptance_count / n_iterations
        return (
            theta_ls,
            plaq_ls,
            acceptance_rate,
            topological_charges,
            hamiltonians,
        )


# %%
