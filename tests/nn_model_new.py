# %%
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

class FieldTransformNet(nn.Module):
    def __init__(self, lattice_size, hidden_size=512):
        super().__init__()
        self.input_size = 2 * lattice_size * lattice_size
        
        # Simplified network with layer normalization instead of batch normalization
        self.layers = nn.ModuleList([
            nn.Linear(self.input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, self.input_size)
        ])
        
        # Initialize weights using Xavier initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        identity = x
        for i in range(0, len(self.layers), 3):
            if i < len(self.layers) - 1:
                x = self.layers[i + 2](self.layers[i + 1](self.layers[i](x)))
            else:
                x = self.layers[i](x)
        return x + identity  # Residual connection

class NeuralTransformation:
    def __init__(self, lattice_size, beta, device='cuda', learning_rate=1e-4):
        self.lattice_size = lattice_size
        self.beta = beta
        self.device = torch.device(device)
        self.model = FieldTransformNet(lattice_size).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

    def forward(self, x):
        """Forward pass through the model"""
        return self.model(x)

    def __call__(self, theta):
        """Make the class callable, implementing the field transformation"""
        with torch.no_grad():
            if len(theta.shape) == 3:  # Add batch dimension if needed
                theta = theta.unsqueeze(0)
            return self.transform_field(theta).squeeze(0)

    def compute_action(self, theta):
        """Compute U(1) action with improved efficiency"""
        theta0, theta1 = theta[:, 0], theta[:, 1]
        theta_P = (
            theta0
            + torch.roll(theta1, shifts=-1, dims=1)
            - torch.roll(theta0, shifts=-1, dims=2)
            - theta1
        )
        return -self.beta * torch.sum(torch.cos(theta_P), dim=(1,2))

    def compute_log_det_jacobian(self, theta_new):
        """Compute log determinant of Jacobian matrix"""
        batch_size = theta_new.shape[0]
        input_size = 2 * self.lattice_size * self.lattice_size
        theta_flat = theta_new.reshape(batch_size, -1)
        
        log_dets = []
        for i in range(batch_size):
            theta_i = theta_flat[i:i+1]  # Keep batch dimension
            jac = torch.autograd.functional.jacobian(self.forward, theta_i)  # Use forward instead
            jac = jac.reshape(input_size, input_size)
            log_det = torch.slogdet(jac)[1]
            log_dets.append(log_det)
        
        return torch.stack(log_dets)

    def transform_field(self, theta_new):
        """Transform field with shape preservation"""
        shape = theta_new.shape
        theta_flat = theta_new.reshape(shape[0], -1)
        theta_orig = self.forward(theta_flat)  # Use forward instead of direct call
        return theta_orig.reshape(shape)

    def compute_transformed_action(self, theta_new):
        """Compute action including Jacobian term"""
        theta_orig = self.transform_field(theta_new)
        orig_action = self.compute_action(theta_orig)
        log_det = self.compute_log_det_jacobian(theta_new)
        return orig_action - log_det

    def compute_force(self, theta, transformed=False):
        """Vectorized force computation"""
        # Ensure input requires gradients
        if not theta.requires_grad:
            theta = theta.clone().requires_grad_(True)
            
        # Compute appropriate action
        if transformed:
            action = self.compute_transformed_action(theta)
        else:
            action = self.compute_action(theta)
            
        # Compute gradients
        force = torch.autograd.grad(
            action.sum(),
            theta,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        return force

    def train_step(self, theta_new):
        """Enhanced training step with gradient clipping"""
        # Ensure input requires gradients
        theta_new = theta_new.requires_grad_(True)
        
        # Compute forces
        force_original = self.compute_force(theta_new, transformed=False)
        force_transformed = self.compute_force(theta_new, transformed=True)
        
        # Combined loss with L2 and L-infinity norms
        loss = (
            torch.norm(force_transformed - force_original, p=2) +
            torch.norm(force_transformed - force_original, p=float('inf'))
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        return loss.item()

    def train(self, n_iterations, batch_size=64, verbose=True):
        """Improved training loop with progress reporting"""
        self.model.train()
        loss_history = []
        best_loss = float('inf')
        patience_counter = 0
        
        pbar = tqdm(range(n_iterations), disable=not verbose)
        for epoch in pbar:
            theta_new = torch.empty(
                (batch_size, 2, self.lattice_size, self.lattice_size),
                device=self.device
            ).uniform_(-math.pi, math.pi)
            
            loss = self.train_step(theta_new)
            loss_history.append(loss)
            
            self.scheduler.step(loss)
            
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 50:
                if verbose:
                    print("Early stopping triggered!")
                break
                
            if verbose and epoch % 100 == 0:
                pbar.set_description(f"Loss: {loss:.4f}")
        
        return loss_history

# %%
