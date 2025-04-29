# %%
import torch
from tqdm import tqdm
from utils import plaq_from_field, plaq_mean_from_field, regularize, topo_from_field


class HMC_U1_FT:
    def __init__(
        self,
        lattice_size,
        beta,
        n_thermalization_steps,
        n_steps,
        step_size,
        field_transformation,
        compute_jac_logdet,
        device="cpu",
        if_tune_step_size=True,
    ):
        """
        Initialize the HMC_U1_FT class.

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
        field_transformation : callable
            The field transformation function that transforms theta_new to theta_ori.
        device : str
            The device to use for computation ('cpu' or 'cuda').
        """
        self.lattice_size = lattice_size
        self.beta = beta
        self.n_thermalization_steps = n_thermalization_steps
        self.n_steps = n_steps
        self.dt = step_size
        self.field_transformation = field_transformation
        self.compute_jac_logdet = compute_jac_logdet
        self.device = torch.device(device)
        self.if_tune_step_size = if_tune_step_size
        
        # Set default data type and device
        torch.set_default_dtype(torch.float32)
        torch.set_default_device(self.device)
        torch.manual_seed(1331)

    def initialize(self):
        """
        Initialize the field configuration to zeros.

        Returns:
        --------
        torch.Tensor
            The initial field configuration.
        """
        return torch.zeros([2, self.lattice_size, self.lattice_size])
    
    def original_action(self, theta):
        """
        Compute the action without field transformation.

        Parameters:
        -----------
        theta : torch.Tensor
            The field configuration.

        Returns:
        --------
        torch.Tensor
            The action value.
        """
        theta_P = plaq_from_field(theta)
        thetaP_wrapped = regularize(theta_P)
        action_value = (-self.beta) * torch.sum(torch.cos(thetaP_wrapped))
        
        # Check if action_value is a scalar
        assert action_value.dim() == 0, "Action value is not a scalar."

        return action_value

    
    def new_action(self, theta_new):
        """
        Compute the transformed action with the Jacobian term.
        
        Parameters:
        -----------
        theta_new : torch.Tensor
            The new field configuration before transformation.

        Returns:
        --------
        torch.Tensor
            The transformed action value.
        """
        theta_ori = self.field_transformation(theta_new)
        original_action_val = self.original_action(theta_ori)

        jacobian_log_det = self.compute_jac_logdet(theta_new.unsqueeze(0)) # [1, 2, L, L]
        jacobian_log_det = jacobian_log_det.squeeze(0)

        new_action_val = original_action_val - jacobian_log_det

        assert new_action_val.dim() == 0, "Transformed action value is not a scalar."

        return new_action_val
    

    def new_force(self, theta_new):
        """
        Compute the force for the HMC update.

        Parameters:
        -----------
        theta_new : torch.Tensor
            The new field configuration after transformation.

        Returns:
        --------
        torch.Tensor
            The force.
        """
        theta_new.requires_grad_(True)
        action_value = self.new_action(theta_new)
        action_value.backward(retain_graph=True)
        ff = theta_new.grad
        theta_new.requires_grad_(False)
        return ff

    def leapfrog(self, theta, pi):
        """
        Perform the leapfrog integration step.

        Parameters:
        -----------
        theta : torch.Tensor
            The initial field configuration.
        pi : torch.Tensor
            The initial momentum.

        Returns:
        --------
        tuple
            The updated field configuration and momentum.
        """
        dt = self.dt
        theta_ = theta + 0.5 * dt * pi
        pi_ = pi - dt * self.new_force(theta_)
        for _ in range(self.n_steps - 1):
            theta_ = theta_ + dt * pi_
            pi_ = pi_ - dt * self.new_force(theta_)
        theta_ = theta_ + 0.5 * dt * pi_
        theta_ = regularize(theta_)
        return theta_, pi_

    def metropolis_step(self, theta):
        """
        Perform a Metropolis step.

        Parameters:
        -----------
        theta : torch.Tensor
            The current field configuration.

        Returns:
        --------
        tuple
            The updated field configuration, acceptance flag, and Hamiltonian value.
        """
        pi = torch.randn_like(theta, device=self.device)
        action_value = self.new_action(theta) 
        H_old = action_value + 0.5 * torch.sum(pi**2)

        new_theta, new_pi = self.leapfrog(theta.clone(), pi.clone())
        new_action_value = self.new_action(new_theta) 
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
        Perform thermalization steps to equilibrate the system.

        Returns:
        --------
        tuple
            The final field configuration, list of plaquette values, and acceptance rate.
        """
        
        # Initial thermalization to get away from cold start
        theta = self.initialize()
        n_initial_therm = self.n_thermalization_steps
        
        print(">>> Initial thermalization...")
        for _ in tqdm(range(n_initial_therm), desc="Initial thermalization"):
            theta, _, _ = self.metropolis_step(theta)
        
        # Tune step size before thermalization
        if self.if_tune_step_size:
            print("Tuning step size before thermalization...")
            self.tune_step_size(theta=theta)
        else:
            print(f"Using step size: {self.dt:.2f}")
        
        theta_new = self.initialize()
        plaq_ls = []
        acceptance_count = 0

        for _ in tqdm(range(self.n_thermalization_steps), desc="Thermalizing"):
            theta_new = regularize(theta_new)
            theta_ori = self.field_transformation(theta_new)
            theta_ori = regularize(theta_ori)
            plaq = plaq_mean_from_field(theta_ori).item()
            theta_new, accepted, _ = self.metropolis_step(theta_new)
            
            plaq_ls.append(plaq)
            if accepted:
                acceptance_count += 1

        acceptance_rate = acceptance_count / self.n_thermalization_steps
        return theta_new, plaq_ls, acceptance_rate

    def run(self, n_iterations, theta, store_interval=1):
        """
        Run the HMC simulation.

        Parameters:
        -----------
        n_iterations : int
            Number of HMC iterations to run.
        theta : torch.Tensor
            Initial field configuration.
        store_interval : int, optional
            Store results every store_interval iterations to save memory (default is 1).

        Returns:
        --------
        tuple
            The final field configuration, list of plaquette values, acceptance rate,
            list of topological charges, and list of Hamiltonian values.
        """

        theta_ori_ls = []
        plaq_ls = []
        hamiltonians = []
        acceptance_count = 0
        topological_charges = []

        for i in tqdm(range(n_iterations), desc="Running HMC"):
            theta, accepted, H_val = self.metropolis_step(theta)
            
            
            if i % store_interval == 0:  # Only store data at specific intervals
                theta = regularize(theta)
                theta_ori = self.field_transformation(theta) 
                theta_ori = regularize(theta_ori)
                theta_ori_ls.append(theta_ori)
                plaq = plaq_mean_from_field(theta_ori).item()
                plaq_ls.append(plaq)
                hamiltonians.append(H_val)
                topological_charges.append(topo_from_field(theta_ori).item())

            if accepted:
                acceptance_count += 1

        acceptance_rate = acceptance_count / n_iterations
        return (
            theta_ori_ls,
            plaq_ls,
            acceptance_rate,
            topological_charges,
            hamiltonians,
        )
