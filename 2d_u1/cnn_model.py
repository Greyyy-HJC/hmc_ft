import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.autograd.functional as F
import numpy as np
from tqdm import tqdm

from utils import plaq_from_field

class StableCNN(nn.Module):
    """Simple CNN model with GELU activation and tanh output scaling"""
    def __init__(self, input_channels=20, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 2, 3, padding=1)
        )

    def forward(self, x):
        return torch.pi * torch.tanh(self.net(x))

def get_plaq_features(plaqphase, device):
    """
    Compute sin/cos features of plaquette phase angles and their neighbors.
    Input: plaqphase with shape [batch_size, L, L]
    Output: features with shape [batch_size, 20, L, L]
    """
    batch_size, L = plaqphase.shape[0], plaqphase.shape[-1]
    features = torch.zeros((batch_size, 20, L, L), device=device)
    
    # Compute features for each direction
    for mu in [0, 1]:
        # Define shifts for this direction
        if mu == 0:
            shifts = [(-1,0), (-1,-1), (1,0), (1,-1)]
        else:
            shifts = [(-1,1), (-1,-1), (0,1), (0,-1)]
            
        # Get neighbor contributions
        angles = [torch.roll(plaqphase, shifts=s, dims=(1,2)) for s in shifts]
        
        # Add sum terms
        if mu == 0:
            angles.append(plaqphase + torch.roll(plaqphase, shifts=(0,-1), dims=(1,2)))
        else:
            angles.append(plaqphase + torch.roll(plaqphase, shifts=(-1,0), dims=(1,2)))
            
        # Compute sin/cos features
        offset = mu * 10
        for idx, angle in enumerate(angles):
            features[:, offset + 2*idx] = torch.cos(angle)
            features[:, offset + 2*idx + 1] = torch.sin(angle)
    
    return features

class FieldTransformation:
    """Neural network based field transformation"""
    def __init__(self, lattice_size, epsilon=0.05, device='cpu'):
        self.L = lattice_size
        self.device = torch.device(device)
        self.epsilon = epsilon
        
        self.model = StableCNN().to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
    def compute_K1(self, theta):
        """
        Compute K1 term for field transformation
        Input: theta with shape [batch_size, 2, L, L]
        Output: K1 with shape [batch_size, 2, L, L]
        """
        batch_size = theta.shape[0]
        plaqphase = torch.zeros((batch_size, self.L, self.L), device=self.device)
        
        for i in range(batch_size):
            plaqphase[i] = plaq_from_field(theta[i])
        
        features = get_plaq_features(plaqphase, self.device)
        return self.model(features)
    
    def forward(self, theta):
        """Transform theta_new to theta_ori"""
        if len(theta.shape) == 3:  # Add batch dimension if needed
            theta = theta.unsqueeze(0)
        return theta + self.epsilon * self.compute_K1(theta)
    
    def inverse(self, theta):
        """Transform theta_ori to theta_new"""
        if len(theta.shape) == 3:  # Add batch dimension if needed
            theta = theta.unsqueeze(0)
        return theta - self.epsilon * self.compute_K1(theta)
    
    def field_transformation(self, theta):
        """
        Field transformation function for HMC.
        Input: theta with shape [2, L, L]
        Output: theta with shape [2, L, L]
        """
        # Add batch dimension for single input
        theta_batch = theta.unsqueeze(0)
        
        # Compute transformation
        K1 = self.compute_K1(theta_batch)
        theta_transformed = theta_batch + self.epsilon * K1
        
        # Remove batch dimension
        return theta_transformed.squeeze(0)
    
    def compute_action(self, theta, beta):
        """
        Compute action for given configuration
        Input: theta with shape [batch_size, 2, L, L] or [2, L, L]
        """
        if len(theta.shape) == 3:
            theta = theta.unsqueeze(0)
        
        batch_size = theta.shape[0]
        total_action = 0
        
        for i in range(batch_size):
            plaq = plaq_from_field(theta[i])
            total_action += torch.sum(torch.cos(plaq))
        
        return -beta * total_action / batch_size  # Average over batch
    
    def compute_force(self, theta, beta, transformed=False):
        """
        Compute force (gradient of action)
        Input: theta with shape [batch_size, 2, L, L] or [2, L, L]
        """
        if len(theta.shape) == 3:
            theta = theta.unsqueeze(0)
        
        batch_size = theta.shape[0]
        theta.requires_grad_(True)
        
        if transformed:
            theta_ori = self.forward(theta)
            action = self.compute_action(theta_ori, beta)
            
            # Compute Jacobian for each sample in the batch
            jac_logdet = 0
            for i in range(batch_size):
                single_theta = theta[i]
                jac = F.jacobian(self.forward, single_theta.unsqueeze(0)).squeeze(0)
                jac_2d = jac.reshape(single_theta.numel(), single_theta.numel())
                jac_logdet += torch.logdet(jac_2d)
            jac_logdet = jac_logdet / batch_size  # Average over batch
            
            total_action = action - jac_logdet
        else:
            total_action = self.compute_action(theta, beta)
            
        force = torch.autograd.grad(total_action, theta, create_graph=True)[0]
        return force.squeeze(0) if len(force.shape) == 4 and force.shape[0] == 1 else force
    
    def train_step(self, theta_ori, beta):
        """Single training step"""
        if len(theta_ori.shape) == 3:
            theta_ori = theta_ori.unsqueeze(0)
        
        theta_new = self.inverse(theta_ori)
        force_ori = self.compute_force(theta_new, beta=2.5)
        force_new = self.compute_force(theta_new, beta, transformed=True)
        
        vol = self.L * self.L
        loss = torch.norm(force_new - force_ori, p=2) / (vol**(1/2)) + \
               torch.norm(force_new - force_ori, p=4) / (vol**(1/4))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, train_data, test_data, beta, n_epochs=100, batch_size=1):
        """Train the model"""
        train_losses = []
        test_losses = []
        
        for epoch in tqdm(range(n_epochs), desc="Training epochs"):
            # Training
            self.model.train()
            indices = torch.randperm(len(train_data))
            epoch_losses = []
            
            # Training loop with progress bar
            train_iter = tqdm(
                range(0, len(train_data), batch_size),
                desc=f"Epoch {epoch+1}/{n_epochs}",
                leave=False
            )
            for i in train_iter:
                batch = train_data[indices[i:i+batch_size]]
                loss = self.train_step(batch, beta)
                epoch_losses.append(loss)
                
                # Update progress bar description with current loss
                train_iter.set_postfix({"Loss": f"{loss:.6f}"})
            
            train_loss = np.mean(epoch_losses)
            
            # Evaluation
            self.model.eval()
            test_losses_epoch = []
            
            # Evaluate without computing gradients
            test_iter = tqdm(
                torch.split(test_data, batch_size),
                desc="Evaluating",
                leave=False
            )
            for batch in test_iter:
                # Forward pass only for evaluation
                if len(batch.shape) == 3:
                    batch = batch.unsqueeze(0)
                    
                theta_new = self.inverse(batch)
                force_ori = self.compute_force(theta_new, beta=2.5)
                force_new = self.compute_force(theta_new, beta, transformed=True)
                
                vol = self.L * self.L
                loss = (torch.norm(force_new - force_ori, p=2) / (vol**(1/2)) + 
                       torch.norm(force_new - force_ori, p=4) / (vol**(1/4))).item()
                
                test_losses_epoch.append(loss)
                test_iter.set_postfix({"Loss": f"{loss:.6f}"})
                
            test_loss = np.mean(test_losses_epoch)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{n_epochs} - "
                  f"Train Loss: {train_loss:.6f} - "
                  f"Test Loss: {test_loss:.6f}")
            
            self.scheduler.step(test_loss)
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train')
        plt.plot(test_losses, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/cnn_loss.pdf', transparent=True)
        plt.show() 