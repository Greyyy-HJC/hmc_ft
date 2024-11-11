# %%
import torch
from tqdm import tqdm
import torch.linalg as linalg
import torch.autograd.functional as F
from utils import plaq_from_field, topo_from_field, plaq_mean_from_field


class HMC_U1_FT:
    def __init__(
        self,
        lattice_size,
        beta,
        n_thermalization_steps,
        n_steps,
        step_size,
        field_transformation,
        jacobian_interval=20,
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
        field_transformation : callable
            The field transformation function.
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
        self.field_transformation = field_transformation
        self.device = torch.device(device)
        self.jacobian_interval = jacobian_interval
        self.jacobian_cache = None

        # Set default data type and device
        torch.set_default_dtype(torch.float32)
        torch.set_default_device(self.device)
        torch.manual_seed(1331)

    def initialize(self):
        return torch.zeros([2, self.lattice_size, self.lattice_size])
    
    def original_action(self, theta):
        """
        Compute the action without field transformation.
        """
        theta_P = plaq_from_field(theta)
        action_value = (-self.beta) * torch.sum(torch.cos(theta_P))
        
        # check if action_value is a scalar
        assert action_value.dim() == 0, "Action value is not a scalar."

        return action_value
    
    def compute_jacobian_log_det(self, theta_new):
        """
        Compute the log determinant of the Jacobian matrix of the transformation.
        
        Parameters:
        -----------
        theta_new : torch.Tensor
            The new field configuration after transformation.
        """
        # Use cached Jacobian if available and not expired        
        if self.step_count % self.jacobian_interval != 0:
            return self.jacobian_cache

        # Compute Jacobian using torch.autograd.functional.jacobian
        jacobian = F.jacobian(self.field_transformation, theta_new)

        # Reshape jacobian to 2D matrix
        jacobian_2d = jacobian.view(-1, jacobian.shape[-1])

        # Compute singular values
        s = linalg.svdvals(jacobian_2d)

        # Compute log determinant as sum of log of singular values
        log_det = torch.sum(torch.log(s))

        # Cache the result
        self.jacobian_cache = log_det

        return log_det
    
    def new_action(self, theta_new):
        """
        Compute the transformed action with the Jacobian term.
        
        Parameters:
        -----------
        theta_new : torch.Tensor
            The new field configuration after transformation.
        """
        theta = self.field_transformation(theta_new)
        original_action_val = self.original_action(theta)

        jacobian_log_det = self.compute_jacobian_log_det(theta_new)

        new_action_val = original_action_val - jacobian_log_det

        assert (
            new_action_val.dim() == 0
        ), "Transformed action value is not a scalar."

        return new_action_val

    def new_force(self, theta_new):
        theta_new.requires_grad_(True)
        action_value = self.new_action(theta_new)
        action_value.backward()
        ff = theta_new.grad
        theta_new.requires_grad_(False)
        return ff

    def leapfrog(self, theta, pi):
        dt = self.dt
        theta_ = theta + 0.5 * dt * pi
        pi_ = pi - dt * self.new_force(theta_)
        for _ in range(self.n_steps - 1):
            theta_ = theta_ + dt * pi_
            pi_ = pi_ - dt * self.new_force(theta_)
        theta_ = theta_ + 0.5 * dt * pi_
        return theta_, pi_

    def metropolis_step(self, theta):
        pi = torch.randn_like(theta, device=self.device)
        action_value = self.new_action(theta)
        H_old = action_value + 0.5 * torch.sum(pi**2)

        new_theta, new_pi = self.leapfrog(theta.clone(), pi.clone())
        new_action_value = self.new_action(new_theta)
        H_new = new_action_value + 0.5 * torch.sum(new_pi**2)

        delta_H = H_new - H_old
        accept_prob = torch.exp(-delta_H)
        
        self.step_count += 1  # Increment step count

        if torch.rand([], device=self.device) < accept_prob:
            return new_theta, True, H_new.item()
        else:
            return theta, False, H_old.item()


    def thermalize(self):
        self.step_count = 0
        theta = self.initialize()
        plaq_ls = []
        acceptance_count = 0

        for _ in tqdm(range(self.n_thermalization_steps), desc="Thermalizing"):
            theta_old = self.field_transformation(theta)
            plaq = plaq_mean_from_field(theta_old).item()
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
        self.step_count = 0
        
        plaq_ls = []
        hamiltonians = []
        acceptance_count = 0
        topological_charges = []

        for i in tqdm(range(n_iterations), desc="Running HMC"):
            theta, accepted, H_val = self.metropolis_step(theta)
            
            if i % store_interval == 0:  # only store data at specific intervals
                theta_old = self.field_transformation(theta)
                plaq = plaq_mean_from_field(theta_old).item()
                plaq_ls.append(plaq)
                hamiltonians.append(H_val)
                topological_charges.append(topo_from_field(theta).item())
                
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





