import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import torch.linalg as linalg
import torch.autograd.functional as F

from utils import plaq_from_field

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
        x_reshaped = x.view(-1, 2, self.lattice_size, self.lattice_size)
        mask1 = torch.zeros_like(x_reshaped)
        mask1[:, 0, 1::2, 1::2] = 1  # Set 1 for odd indices in both dimensions for channel 0

        mask2 = torch.zeros_like(x_reshaped)
        mask2[:, 1, 0::2, 0::2] = 1  # Set 1 for even indices in both dimensions for channel 1

        for mask in [mask1, mask2]:
            y = x_reshaped.clone()  # Ensure correct shape
            y = y * mask
            y = self.conv_layers(y)
            x_reshaped = x_reshaped + y

        return x_reshaped.view(-1, 2 * self.lattice_size * self.lattice_size)  # Flatten
    
class NNFieldTransformation:
    def __init__(self, lattice_size, model_type='CNN', epsilon=0.1, epsilon_decay=1, jacobian_interval=20, device='cpu'):
        self.lattice_size = lattice_size
        self.input_size = 2 * lattice_size * lattice_size
        self.output_size = 2 * lattice_size * lattice_size
        self.device = torch.device(device)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.jacobian_interval = jacobian_interval
        self.jacobian_cache = None

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
        # theta has shape (2, L, L)
        theta_tensor = theta.to(self.device).view(1, -1)

        # Forward pass through the neural network
        delta_theta_tensor = self.model(theta_tensor)

        # Limit the transformation magnitude
        theta_transformed_tensor = theta_tensor + self.epsilon * delta_theta_tensor

        # Reshape back to (2, L, L)
        theta_transformed = theta_transformed_tensor.view(2, self.lattice_size, self.lattice_size)

        # Ensure angles are within [-pi, pi)
        theta_transformed = torch.remainder(theta_transformed + math.pi, 2 * math.pi) - math.pi
        
        return theta_transformed
    
    def field_transformation(self, theta):
        """
        Define a field transformation function. Field transformation transforms the new field back to the original field.
        """
        return self(theta)
    
    def original_action(self, theta, beta):
        """
        Compute the action without field transformation.
        """
        theta_P = plaq_from_field(theta)
        action_value = (-beta) * torch.sum(torch.cos(theta_P))
        
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
    
    def new_action(self, theta_new, beta):
        theta = self.field_transformation(theta_new)
        original_action_val = self.original_action(theta, beta)

        jacobian_log_det = self.compute_jacobian_log_det(theta_new)

        new_action_val = original_action_val - jacobian_log_det

        assert (
            new_action_val.dim() == 0
        ), "Transformed action value is not a scalar."

        return new_action_val
    
    def original_force(self, theta, beta):
        """
        Compute the force (gradient of the action) using PyTorch operations.
        """
        theta.requires_grad_(True)  # Ensure gradients are tracked
        action = self.original_action(theta, beta)
        force = torch.autograd.grad(action, theta, create_graph=True)[0]

        return force

    def new_force(self, theta_new, beta):
        """
        Compute the new force using PyTorch operations.
        """
        theta_new.requires_grad_(True)
        new_action_val = self.new_action(theta_new, beta)
        force = torch.autograd.grad(new_action_val, theta_new, create_graph=True)[0]

        return force
            
    def train(self, beta, n_iterations):
        loss_history = []  # To store loss values

        self.step_count = 0  # Initialize step count

        for _ in tqdm(range(n_iterations), desc="Training Neural Network"):
            # Initialize U with shape (2, L, L), this is new field configuration
            # U_ini = torch.empty((2, self.lattice_size, self.lattice_size), device=self.device).uniform_(-math.pi, math.pi)
            U_ini = torch.zeros([2, self.lattice_size, self.lattice_size])
            U_transformed = self.field_transformation(U_ini)

            # Compute forces (gradients of actions)
            force_original = self.original_force(U_transformed, beta)
            force_new = self.new_force(U_ini, beta)

            # Compute the loss using combined norm
            loss = torch.norm(force_new - force_original, p=2) + torch.norm(force_new - force_original, p=float('inf'))
            # loss = torch.norm(force_new - force_original, p=2)

            # Log the loss
            loss_history.append(loss.item())

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Decay epsilon
            self.epsilon *= self.epsilon_decay

            self.step_count += 1

        # Plot the loss history
        plt.figure(figsize=(9, 6))
        plt.plot(loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.show()
    
