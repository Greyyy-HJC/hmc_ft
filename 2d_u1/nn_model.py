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
    
    def compute_force_torch(self, theta, beta):
        """
        Compute the force (gradient of the action) using PyTorch operations.
        """
        theta.requires_grad_(True)  # Ensure gradients are tracked
        action = self.compute_action_torch(theta, beta)
        force = torch.autograd.grad(action, theta, create_graph=True)[0]
        return force
    
    def metropolis_acceptance(self, delta_H):
        """
        Metropolis-Hastings acceptance step.
        """
        if delta_H < 0:
            return True
        elif torch.rand(1, device=self.device).item() < math.exp(-delta_H):
            return True
        return False
            
    def train(self, beta, n_iterations):
        loss_history = []  # To store loss values

        for _ in tqdm(range(n_iterations), desc="Training Neural Network"):
            # Initialize U with shape (2, L, L)
            U = torch.empty((2, self.lattice_size, self.lattice_size), device=self.device).uniform_(-math.pi, math.pi)
            U_tensor = U.view(1, -1)  # Reshape to (1, 2 * L * L)

            # Forward pass through the neural network
            delta_U_tensor = self.model(U_tensor)
            U_transformed_tensor = U_tensor + self.epsilon * delta_U_tensor
            U_transformed = U_transformed_tensor.view(2, self.lattice_size, self.lattice_size)
            
            # Calculate original and transformed actions
            action_original = self.compute_action_torch(U, beta)
            action_transformed = self.compute_action_torch(U_transformed, beta)

            # Calculate Hamiltonian change and apply Metropolis-Hastings decision
            delta_H = action_transformed.item() - action_original.item()
            if not self.metropolis_acceptance(delta_H):
                continue  # If the new configuration is rejected, skip the optimization step

            # Compute forces (gradients of actions)
            force_original = self.compute_force_torch(U, beta)
            force_transformed = self.compute_force_torch(U_transformed, beta)

            # Compute the loss using combined norm
            loss = torch.norm(force_transformed - force_original, p=2) + torch.norm(force_transformed - force_original, p=float('inf'))

            # Log the loss
            loss_history.append(loss.item())

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Decay epsilon
            self.epsilon *= self.epsilon_decay

        # Plot the loss history
        plt.figure(figsize=(9, 6))
        plt.plot(loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.show()
    
    def compute_action_torch(self, theta, beta):
        """
        Compute the action using PyTorch operations.
        theta: Tensor of shape (2, L, L)
        beta: Inverse coupling constant
        """
        # Extract theta components
        theta0 = theta[0]
        theta1 = theta[1]

        # Compute plaquette angles using periodic boundary conditions
        theta_P = (
            theta0
            + torch.roll(theta1, shifts=-1, dims=0)
            - torch.roll(theta0, shifts=-1, dims=1)
            - theta1
        )

        # Compute the action
        action = -beta * torch.sum(torch.cos(theta_P))
        return action