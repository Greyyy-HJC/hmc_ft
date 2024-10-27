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
        n_steps,
        step_size,
        field_transformation=None,
        device="cpu",
        jacobian_interval=10,
    ):

        self.lattice_size = lattice_size
        self.beta = beta
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
        torch.manual_seed(1331)

    def initialize(self):
        """
        Initialize the field configuration (theta) uniformly between -pi and pi.
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
    
    def force(self, theta):
        theta.requires_grad_(True)
        action_value = self.action(theta)
        action_value.backward()
        ff = theta.grad
        theta.requires_grad_(False)
        return ff
    
    def leapfrog(self, theta, pi):
        theta_ = theta + 0.5*self.dt*pi
        pi_ = pi + (-self.dt)*self.force(theta_)
        for _ in range(self.n_steps-1):
            theta_ = theta_ + self.dt*pi_
            pi_ = pi_ + (-self.dt)*self.force(theta_)
        theta_ = theta_ + 0.5*self.dt*pi_
        return (theta_, pi_)

    def metropolis_step(self, theta):
        """
        Perform a Metropolis step.
        
        Parameters:
        -----------
        theta : torch.Tensor
            Can be the old field configuration without transformation or the new field configuration after transformation.
        """
        pi = torch.randn_like(theta, device=self.device)
        action_value = self.action(theta)
        H_old = action_value + 0.5 * torch.sum(pi**2)

        new_theta, new_pi = self.leapfrog(theta, pi)
        new_action_value = self.action(new_theta)
        H_new = new_action_value + 0.5 * torch.sum(new_pi**2)

        delta_H = H_new - H_old
        accept_prob = torch.exp(-delta_H)

        if torch.rand([], device=self.device) < accept_prob:
            return new_theta, True, H_new.item()
        else:
            return theta, False, H_old.item()
        
    def regularize(self, theta):
        theta_res = (theta - math.pi) / (2*math.pi)
        return 2*math.pi*(theta_res - torch.floor(theta_res) - 0.5)
        
        
    def topological_charge(self, theta):
        theta_P = self.plaqphase(theta)
        return torch.floor(0.1 + torch.sum(self.regularize(theta_P)) / (2*math.pi))

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
            
            action_value = self.action(theta).item()
                
            actions.append(action_value)
            hamiltonians.append(H)
            topological_charges.append(self.topological_charge(theta).item())
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
