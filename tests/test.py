# %%

import torch
import math
from tqdm import tqdm
from torch.autograd import grad
import torch.autograd.functional as F
import torch.linalg as linalg

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
        self.field_transformation = field_transformation  # This should now be the NNFieldTransformation instance
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
        return torch.empty(
            [2, self.lattice_size, self.lattice_size], device=self.device
        ).uniform_(-math.pi, math.pi)

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
        theta = theta.detach().requires_grad_(True)
        
        if self.field_transformation is None:
            action_value = self.action(theta)
        else:
            action_value = self.transformed_action(theta)

        force = grad(action_value, theta, create_graph=False, retain_graph=False)[0]
        return force

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
        pi = pi - 0.5 * dt * self.force(theta)
        for _ in range(self.n_steps):
            theta = theta + dt * pi
            theta = self.regularize(theta)
            pi = pi - dt * self.force(theta)
        theta = theta + 0.5 * dt * pi
        return theta.detach(), pi

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

    def regularize(self, theta):
        """
        Regularize the angle to be within the range [-pi, pi].
        """
        return theta - 2 * math.pi * torch.floor((theta + math.pi) / (2 * math.pi))

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

            # Print progress every 10% of thermalization
            if (_ + 1) % (self.n_thermalization_steps // 10) == 0:
                print(
                    f"Thermalization progress: {(_+1)/self.n_thermalization_steps:.1%}, "
                    f"Current action: {action_value:.2f}"
                )

        acceptance_rate = acceptance_count / self.n_thermalization_steps
        return theta, torch.tensor(actions), acceptance_rate

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
            torch.tensor(actions),
            acceptance_rate,
            torch.tensor(topological_charges),
            torch.tensor(hamiltonians),
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


# %%
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

# Base class for models
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

# Fully connected neural network (SimpleNN)
class SimpleNN(BaseModel):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),  # Hidden layer with 128 neurons
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.layer(x)

# Convolutional neural network (CNNModel)
class CNNModel(BaseModel):
    def __init__(self, lattice_size):
        super(CNNModel, self).__init__()
        self.lattice_size = lattice_size

        # Define convolutional layers, use GELU instead of ReLU
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # Input channels = 2
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1)   # Output channels = 2
        )

    def forward(self, x):
        x = x.view(-1, 2, self.lattice_size, self.lattice_size)  # Ensure correct shape
        x = self.conv_layers(x)  # Apply convolution
        x += x  # local update superposition
        return x.view(-1, 2 * self.lattice_size * self.lattice_size)  # Flatten
    
class NNFieldTransformation:
    def __init__(self, lattice_size, model_type='CNN', epsilon=0.01, epsilon_decay=1, device='cpu'):
        self.lattice_size = lattice_size
        self.input_size = 2 * lattice_size * lattice_size
        self.output_size = 2 * lattice_size * lattice_size
        self.device = torch.device(device)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Choose the model type
        if model_type == 'SimpleNN':
            self.model = SimpleNN(self.input_size, self.output_size)
        elif model_type == 'CNN':
            self.model = CNNModel(lattice_size)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'SimpleNN' or 'CNN'.")

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def __call__(self, theta):
        theta_tensor = theta.to(self.device).view(1, -1)
        delta_theta_tensor = self.model(theta_tensor)
        theta_transformed_tensor = theta_tensor + self.epsilon * delta_theta_tensor
        theta_transformed = theta_transformed_tensor.view(2, self.lattice_size, self.lattice_size)
        theta_transformed = torch.remainder(theta_transformed + math.pi, 2 * math.pi) - math.pi
        return theta_transformed

    def train(self, beta, n_iterations):
        loss_history = []

        for _ in tqdm(range(n_iterations), desc="Training Neural Network"):
            U = torch.empty((2, self.lattice_size, self.lattice_size), device=self.device).uniform_(-math.pi, math.pi)
            U_tensor = U.view(1, -1)

            delta_U_tensor = self.model(U_tensor)
            U_transformed_tensor = U_tensor + self.epsilon * delta_U_tensor
            U_transformed = U_transformed_tensor.view(2, self.lattice_size, self.lattice_size)
            
            action_original = self.compute_action_torch(U, beta)
            action_transformed = self.compute_action_torch(U_transformed, beta)

            delta_H = action_transformed.item() - action_original.item()
            if not self.metropolis_acceptance(delta_H):
                continue

            force_original = self.compute_force_torch(U, beta)
            force_transformed = self.compute_force_torch(U_transformed, beta)

            loss = torch.norm(force_transformed - force_original, p=2) + torch.norm(force_transformed - force_original, p=float('inf'))

            loss_history.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.epsilon *= self.epsilon_decay

        plt.figure(figsize=(9, 6))
        plt.plot(loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.show()

    def compute_action_torch(self, theta, beta):
        theta0 = theta[0]
        theta1 = theta[1]
        theta_P = (
            theta0
            + torch.roll(theta1, shifts=-1, dims=0)
            - torch.roll(theta0, shifts=-1, dims=1)
            - theta1
        )
        action = -beta * torch.sum(torch.cos(theta_P))
        return action

    def compute_force_torch(self, theta, beta):
        theta = theta.requires_grad_(True)
        action = self.compute_action_torch(theta, beta)
        force = torch.autograd.grad(action, theta, create_graph=True)[0]
        return force

    def metropolis_acceptance(self, delta_H):
        if delta_H < 0:
            return True
        elif torch.rand(1, device=self.device).item() < math.exp(-delta_H):
            return True
        return False


# # %%
# # Parameters
# """ 
# total_time = n_steps * step_size should be around 1 to 2 units.
# If the energy conservation is not good, try to reduce the step size.
# """

# lattice_size = 16
# volume = lattice_size ** 2
# beta = 6
# n_thermalization_steps = 100
# n_steps = 100
# step_size = 0.1
# n_iterations = 200

# # Initialize device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Set default type to float (float32)
# torch.set_default_dtype(torch.float32)


# # %%
# print(">>> No Field Transformation")

# # Initialize HMC
# hmc = HMC_U1(lattice_size, beta, n_thermalization_steps, n_steps, step_size, device=device)

# # Thermalize the system
# theta_thermalized, thermalization_actions, thermalization_acceptance_rate = hmc.thermalize()

# # Run HMC without field transformation
# final_config, actions, acceptance_rate, topological_charges, hamiltonians = hmc.run(n_iterations, theta_thermalized)



# # %%
# print(">>> Neural Network Field Transformation")

# # Train the neural network force
# nn_transformation = NNFieldTransformation(lattice_size, model_type='CNN', device=device)
# nn_transformation.train(beta, n_iterations=500)







# %%

