# %%
import torch
import math
from tqdm import tqdm
from torch.autograd import grad
import torch.autograd.functional as F
import torch.linalg as linalg

#! Note the field transformation should transform new field to old field, where the old field is the one we want to sample with Wilson action.

class HMC_U1:
    def __init__(
        self,
        lattice_size,
        beta,
        n_thermalization_steps,
        n_steps,
        step_size,
        field_transformation=None,
        device="cpu",
        jacobian_interval=10,
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
        field_transformation : function or None
            A function that applies the field transformation. If None, no transformation is applied. Note the field transformation should transform new field to old field.
        device : str
            The device to use for computation ('cpu' or 'cuda').
        jacobian_interval : int
            The interval at which to compute the Jacobian log determinant.
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
        self.step_count = 0

        # Set default data type and device
        torch.set_default_dtype(torch.float64)
        torch.set_default_device(self.device)
        torch.manual_seed(2048)

    def initialize(self):
        """
        Initialize the field configuration (theta) zero.
        """
        return torch.zeros([2, self.lattice_size, self.lattice_size])

    def plaqphase(self, theta):
        """
        Compute the plaquette phase.
        """
        theta0, theta1 = theta[0], theta[1]
        thetaP = theta0 - theta1 - torch.roll(theta0, shifts=-1, dims=1) + torch.roll(theta1, shifts=-1, dims=0)
        
        return thetaP

    def action(self, theta):
        """
        Compute the action without field transformation.
        
        Parameters:
        -----------
        theta : torch.Tensor
            The old field configuration without transformation.
        """
        theta_P = self.plaqphase(theta)  # calculate plaquette phase
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

    def transformed_action(self, theta_new):
        """
        Compute the transformed action with the Jacobian term.
        
        Parameters:
        -----------
        theta_new : torch.Tensor
            The new field configuration after transformation.
        """
        theta = self.field_transformation(theta_new)
        original_action = self.action(theta)

        jacobian_log_det = self.compute_jacobian_log_det(theta_new)

        transformed_action_value = original_action - jacobian_log_det

        assert (
            transformed_action_value.dim() == 0
        ), "Transformed action value is not a scalar."

        return transformed_action_value

    def force(self, theta):
        """
        Compute the force for the current field configuration.
        
        Parameters:
        -----------
        theta : torch.Tensor
            Can be the old field configuration without transformation or the new field configuration after transformation.
        """
        theta.requires_grad_(True)
        
        if self.field_transformation is None:
            action_value = self.action(theta)
        else:
            action_value = self.transformed_action(theta)

        action_value.backward()
        ff = theta.grad
        theta.requires_grad_(False)
        return ff

    def leapfrog(self, theta, pi):
        """
        Perform leapfrog integration.
        
        Parameters:
        -----------
        theta : torch.Tensor
            Can be the old field configuration without transformation or the new field configuration after transformation.
        pi : torch.Tensor
            The momentum.
        """
        dt = self.dt
        theta_ = theta + 0.5 * dt * pi
        pi_ = pi - dt * self.force(theta_)
        for _ in range(self.n_steps):
            theta_ = theta_ + dt * pi_
            pi_ = pi_ - dt * self.force(theta_)
        theta_ = theta_ + 0.5 * dt * pi_
        return theta_, pi_

    def metropolis_step(self, theta):
        """
        Perform a Metropolis step.
        
        Parameters:
        -----------
        theta : torch.Tensor
            Can be the old field configuration without transformation or the new field configuration after transformation.
        """
        self.step_count += 1  # Increment step count
        pi = torch.randn_like(theta, device=self.device)
        
        if self.field_transformation is None:
            action_value = self.action(theta)
        else:
            action_value = self.transformed_action(theta)
        
        H_old = action_value + 0.5 * torch.sum(pi**2)

        new_theta, new_pi = self.leapfrog(theta.clone(), pi.clone())
        
        if self.field_transformation is None:
            new_action_value = self.action(new_theta)
        else:
            new_action_value = self.transformed_action(new_theta)
        
        H_new = new_action_value + 0.5 * torch.sum(new_pi**2)

        delta_H = H_new - H_old
        accept_prob = torch.exp(-delta_H)

        if torch.rand([], device=self.device) < accept_prob:
            return new_theta, True, H_new.item()
        else:
            return theta, False, H_old.item()

    # def regularize(self, theta):
    #     """
    #     Regularize the angle to be within the range [-pi, pi].
    #     """
    #     return theta - 2 * math.pi * torch.floor((theta + math.pi) / (2 * math.pi))
    
    def regularize(self, theta):
        """
        Regularize the angle to be within the range [-pi, pi].
        """
        theta_res = (theta - math.pi) / (2*math.pi)
        return (2*math.pi) * (theta_res - torch.floor(theta_res) - 0.5)

    def thermalize(self):
        """
        Thermalize the system.
        
        Returns:
        --------
        theta : torch.Tensor
            The field configuration after thermalization.
        actions : list
            The list of actions during thermalization.
        acceptance_rate : float
            The acceptance rate during thermalization.
        """
        theta = self.initialize()
        actions = []
        acceptance_count = 0

        for _ in tqdm(range(self.n_thermalization_steps), desc="Thermalizing"):
            if self.field_transformation is None:
                action_value = self.action(theta).item()
            else:
                action_value = self.transformed_action(theta).item()
                
            theta, accepted, _ = self.metropolis_step(theta)
            
            actions.append(action_value)
            if accepted:
                acceptance_count += 1

        acceptance_rate = acceptance_count / self.n_thermalization_steps
        return theta, actions, acceptance_rate

    def run(self, n_iterations, theta):
        """
        Run HMC sampling.
        """
        actions = []
        hamiltonians = []
        acceptance_count = 0
        topological_charges = []

        for _ in tqdm(range(n_iterations), desc="Running HMC"):
            theta, accepted, H = self.metropolis_step(theta)
            
            if self.field_transformation is None:
                action_value = self.action(theta).item()
            else:
                action_value = self.transformed_action(theta).item()
                
            actions.append(action_value)
            hamiltonians.append(H)
            topological_charges.append(self.topological_charge(theta))
            if accepted:
                acceptance_count += 1

        acceptance_rate = acceptance_count / n_iterations
        return (
            theta,
            actions,
            acceptance_rate,
            topological_charges,
            hamiltonians,
        )

    def topological_charge(self, theta):
        """
        Compute the topological charge.
        """
        theta_P = self.plaqphase(theta)
        theta_P_wrapped = self.regularize(theta_P)
        # add 0.1 to avoid round-off error
        Q = torch.floor(0.1 + torch.sum(theta_P_wrapped) / (2 * math.pi))
        return Q.item()

