import torch
import torch.nn as nn
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import torch.linalg as linalg
import torch.autograd.functional as F

from utils import plaq_from_field, regularize

class StableCNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, hidden_channels=64, kernel_size=3):
        super(StableCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, output_channels, kernel_size, padding=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # Scale output to [-pi, pi] range using tanh
        x = torch.pi * torch.tanh(x)
        return x


def compute_neighbor_contributions(plaqphase, L, device):
    neighbor_contributions = torch.zeros((2, 5, L, L), dtype=torch.float32, device=device)

    # Use tensor slicing and rolling to compute neighbors
    neighbor_contributions[0, 0] = torch.roll(plaqphase, shifts=(-1, 0), dims=(0, 1))  # (x-1, y)
    neighbor_contributions[0, 1] = torch.roll(plaqphase, shifts=(-1, -1), dims=(0, 1))  # (x-1, y-1)
    neighbor_contributions[0, 2] = torch.roll(plaqphase, shifts=(1, 0), dims=(0, 1))  # (x+1, y)
    neighbor_contributions[0, 3] = torch.roll(plaqphase, shifts=(1, -1), dims=(0, 1))  # (x+1, y-1)
    neighbor_contributions[0, 4] = plaqphase + torch.roll(plaqphase, shifts=(0, -1), dims=(0, 1))  # (x, y) + (x, y-1)

    neighbor_contributions[1, 0] = torch.roll(plaqphase, shifts=(-1, 1), dims=(0, 1))  # (x-1, y+1)
    neighbor_contributions[1, 1] = torch.roll(plaqphase, shifts=(-1, -1), dims=(0, 1))  # (x-1, y-1)
    neighbor_contributions[1, 2] = torch.roll(plaqphase, shifts=(0, 1), dims=(0, 1))  # (x, y+1)
    neighbor_contributions[1, 3] = torch.roll(plaqphase, shifts=(0, -1), dims=(0, 1))  # (x, y-1)
    neighbor_contributions[1, 4] = plaqphase + torch.roll(plaqphase, shifts=(-1, 0), dims=(0, 1))  # (x, y) + (x-1, y)

    angle_input = torch.zeros((2, 10, L, L), dtype=torch.float32, device=device)

    angle_input[0, 0] = torch.cos(neighbor_contributions[0, 0])
    angle_input[0, 1] = torch.sin(neighbor_contributions[0, 0])
    angle_input[0, 2] = torch.cos(neighbor_contributions[0, 1])
    angle_input[0, 3] = torch.sin(neighbor_contributions[0, 1])
    angle_input[0, 4] = torch.cos(neighbor_contributions[0, 2])
    angle_input[0, 5] = torch.sin(neighbor_contributions[0, 2])
    angle_input[0, 6] = torch.cos(neighbor_contributions[0, 3])
    angle_input[0, 7] = torch.sin(neighbor_contributions[0, 3])
    angle_input[0, 8] = torch.cos(neighbor_contributions[0, 4])
    angle_input[0, 9] = torch.sin(neighbor_contributions[0, 4])

    angle_input[1, 0] = torch.cos(neighbor_contributions[1, 0])
    angle_input[1, 1] = torch.sin(neighbor_contributions[1, 0])
    angle_input[1, 2] = torch.cos(neighbor_contributions[1, 1])
    angle_input[1, 3] = torch.sin(neighbor_contributions[1, 1])
    angle_input[1, 4] = torch.cos(neighbor_contributions[1, 2])
    angle_input[1, 5] = torch.sin(neighbor_contributions[1, 2])
    angle_input[1, 6] = torch.cos(neighbor_contributions[1, 3])
    angle_input[1, 7] = torch.sin(neighbor_contributions[1, 3])
    angle_input[1, 8] = torch.cos(neighbor_contributions[1, 4])
    angle_input[1, 9] = torch.sin(neighbor_contributions[1, 4])

    return angle_input
    # return neighbor_contributions


class NNFieldTransformation:
    def __init__(self, lattice_size, epsilon=0.01, jacobian_interval=20, device='cpu'):
        self.lattice_size = lattice_size
        self.device = torch.device(device)
        self.epsilon = epsilon
        self.jacobian_interval = jacobian_interval

        # Initialize CNN model
        self.model = StableCNN(input_channels=10)
        self.model.to(self.device)

    def __call__(self, theta_new):
        L = self.lattice_size

        plaqphase = plaq_from_field(theta_new)
        plaqphase = regularize(plaqphase)

        # Collect neighborhood features for CNN input
        neighbor_contributions = compute_neighbor_contributions(plaqphase, L, self.device)

        if torch.isnan(neighbor_contributions).any():
            print("neighbor_contributions contains NaN values!")

        # Pass neighbor contributions through CNN
        K1 = torch.zeros((2, L, L), dtype=torch.float32, device=self.device)
        for mu in range(2):
            neighbor_input = neighbor_contributions[mu].unsqueeze(0)

            K1[mu] = self.model(neighbor_input).squeeze()

        if torch.isnan(K1).any():
            print("K1 contains NaN values!")

        # Apply the transformation
        theta_trans = theta_new + K1 * self.epsilon
        return theta_trans

    
    def field_transformation(self, theta_new):
        """
        Define a field transformation function. Field transformation transforms the new field back to the original field.
        """
        # return regularize( self(theta_new) )
        return self(theta_new)
    
    def original_action(self, theta_ori, beta):
        """
        Compute the action without field transformation.
        """
        theta_P = plaq_from_field(theta_ori)
        theta_wrapped = regularize(theta_P)
        action_value = (-beta) * torch.sum(torch.cos(theta_wrapped))
        
        # check if action_value is a scalar
        assert action_value.dim() == 0, "Action value is not a scalar."

        return action_value
    
    def compute_jacobian_log_det(self, theta_new):
        """
        Compute the log determinant of the Jacobian matrix of the transformation.

        field_transformation(theta_new) = theta_ori
        
        Parameters:
        -----------
        theta_new : torch.Tensor
            The new field configuration after transformation.

        Returns:
        --------
        torch.Tensor
            The log determinant of the Jacobian matrix.
        """
        # Use cached Jacobian if available and not expired        
        if self.step_count % self.jacobian_interval != 0:
            return self.jacobian_cache

        # Compute Jacobian using torch.autograd.functional.jacobian
        jacobian = F.jacobian(self.field_transformation, theta_new)

        # Reshape jacobian to 2D matrix
        jacobian_2d = jacobian.reshape(theta_new.numel(), theta_new.numel())
        log_det = torch.logdet(jacobian_2d)
        
        if torch.isnan(log_det) or torch.isinf(log_det):
            print(">>> Warning: Invalid values detected of the log det Jacobian!")

        # Cache the result
        self.jacobian_cache = log_det

        return log_det

    
    def new_action(self, theta_new, beta):
        theta_ori = self.field_transformation(theta_new)
        original_action_val = self.original_action(theta_ori, beta)
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

        if torch.isnan(force).any():
            print("Original force contains NaN!")

        return force

    def new_force(self, theta_new, beta):
        """
        Compute the new force using PyTorch operations.
        """
        theta_new.requires_grad_(True)
        new_action_val = self.new_action(theta_new, beta)
        force = torch.autograd.grad(new_action_val, theta_new, create_graph=True)[0]

        if torch.isnan(force).any():
            print("New force contains NaN!")

        return force
            
    def train(self, beta, n_epochs=100):
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        loss_history = []  # To store loss values
        self.step_count = 0  # Initialize step count

        for epoch in tqdm(range(n_epochs), desc="Training Neural Network"):
            # Initialize U with shape (2, L, L), this is new field configuration
            theta_new = torch.empty((2, self.lattice_size, self.lattice_size), device=self.device).uniform_(-math.pi, math.pi)

            # Compute forces (gradients of actions)
            # force_original = self.original_force(theta_new, beta=2.5) # so that to make the new force smaller

            #todo
            theta_ori = self.field_transformation(theta_new)
            theta_ori = regularize(theta_ori)
            force_original = self.original_force(theta_ori, beta)

            force_new = self.new_force(theta_new, beta)

            # Compute the loss using combined norm
            vol = self.lattice_size ** 2
            loss = torch.norm(force_new - force_original, p=2) #/ (vol ** (1/2)) + torch.norm(force_new - force_original, p=4) / (vol ** (1/4))

            if torch.isnan(loss).any():
                print("Loss is NaN!")

            # Log the loss
            loss_history.append(loss.item())

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item()) 

            # Check for NaN in model parameters
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    print(f"Epoch {epoch}, {name} contains NaN!")

            self.step_count += 1

        # Plot the loss history
        plt.figure(figsize=(6, 3.5))
        plt.plot(loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.tight_layout()
        plt.show()
    
