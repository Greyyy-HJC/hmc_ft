import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

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

        # Define convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # Input channels = 2 (directions), output channels = 32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1)   # Output channels = 2 to match input channels
        )

    def forward(self, x):
        # Reshape input to (batch_size, channels, height, width)
        x = x.view(-1, 2, self.lattice_size, self.lattice_size)
        x = self.conv_layers(x)
        # Flatten output back to (batch_size, 2 * L * L)
        x = x.view(-1, 2 * self.lattice_size * self.lattice_size)
        return x

class NNFieldTransformation:
    def __init__(self, lattice_size, model_type='CNN'):
        self.lattice_size = lattice_size
        self.input_size = 2 * lattice_size * lattice_size  # Adjusted input size
        self.output_size = 2 * lattice_size * lattice_size  # Adjusted output size
        self.device = torch.device('cpu')  # Change to 'cuda' if using GPU

        # Choose the model type
        if model_type == 'SimpleNN':
            self.model = SimpleNN(self.input_size, self.output_size)
        elif model_type == 'CNN':
            self.model = CNNModel(lattice_size)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'SimpleNN' or 'CNN'.")

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.to(self.device)

    def __call__(self, U):
        # U has shape (2, L, L)
        U_tensor = torch.tensor(U, dtype=torch.float32, device=self.device)
        U_tensor = U_tensor.view(1, -1)  # Add batch dimension: shape (1, 2 * L * L)

        # Forward pass through the neural network
        delta_U_tensor = self.model(U_tensor)

        # Limit the transformation magnitude
        epsilon = 0.01  # Scaling factor to limit transformation magnitude
        U_transformed_tensor = U_tensor + epsilon * delta_U_tensor

        # Reshape back to (2, L, L)
        U_transformed = U_transformed_tensor.detach().cpu().numpy().reshape(U.shape)

        # Ensure angles are within [0, 2π)
        U_transformed = np.mod(U_transformed, 2 * np.pi)
        return U_transformed

    def train(self, hmc_instance, n_iterations):
        for _ in tqdm(range(n_iterations), desc="Training Neural Network"):
            # Initialize U using the HMC instance
            U = hmc_instance.initialize()  # U shape: (2, L, L)

            # Convert U to tensor
            U_tensor = torch.tensor(U, dtype=torch.float32, device=self.device)
            U_tensor = U_tensor.view(1, -1)  # Shape: (1, 2 * L * L)

            # Forward pass through the neural network
            delta_U_tensor = self.model(U_tensor)

            # Limit the transformation magnitude
            epsilon = 0.01  # Scaling factor to limit transformation magnitude
            U_transformed_tensor = U_tensor + epsilon * delta_U_tensor

            # Reshape tensors back to (2, L, L)
            U_transformed = U_transformed_tensor.view(2, self.lattice_size, self.lattice_size)

            # Compute the action using PyTorch functions
            action_original = self.compute_action_torch(U_tensor.view(2, self.lattice_size, self.lattice_size), hmc_instance)
            action_transformed = self.compute_action_torch(U_transformed, hmc_instance)

            # Compute the transformation strength
            transformation_strength = torch.mean((U_transformed - U_tensor.view(2, self.lattice_size, self.lattice_size))**2)
            lambda_reg = 1.0  # Regularization strength
            loss = (action_transformed - action_original) + lambda_reg * transformation_strength

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_action_torch(self, theta, hmc_instance):
        """
        Compute the action using PyTorch operations.
        theta: Tensor of shape (2, L, L)
        """
        beta = hmc_instance.beta

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