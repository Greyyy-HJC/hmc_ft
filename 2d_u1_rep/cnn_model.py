# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import get_musk, plaq_from_field_batch
    
class SimpleCNN(nn.Module):
    """Simple CNN model with GELU activation"""
    def __init__(self, input_channels=2, output_channels=2, kernel_size=(3, 2)):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, 
            output_channels, 
            kernel_size, 
            padding='same',  # Use 'same' padding to maintain input size
            padding_mode='circular'  # Use circular padding to maintain periodic boundary conditions
        )
        self.activation = nn.GELU()

    def forward(self, x):
        # input shape: [batch_size, 2, L, L]
        x = self.conv(x)
        x = self.activation(x)
        x = torch.arctan(x) / torch.pi * 2 # range [-1, 1]
        # output shape: [batch_size, 2, L, L]
        return x 

    

class FieldTransformation:
    """Neural network based field transformation"""
    def __init__(self, lattice_size, device='cpu'):
        self.L = lattice_size
        self.device = torch.device(device)
        
        self.model = SimpleCNN().to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
    def compute_K0(self, theta):
        """
        Compute K0 for given theta
        Input: theta with shape [batch_size, 2, L, L]
        Output: K0 with shape [batch_size, 2, L, L]
        """
        batch_size = theta.shape[0]
        K0 = torch.zeros((batch_size, 2, self.L, self.L), device=self.device)
        
        # Compute plaquettes for all batches at once
        plaq = plaq_from_field_batch(theta)  # [batch_size, L, L]
        
        for index in range(1):
            field_musk, plaq_musk = get_musk(index, batch_size, self.L)
            field_musk = field_musk.to(self.device)
            plaq_musk = plaq_musk.to(self.device)
            
            # Apply mask and compute features for all batches at once
            plaq_masked = plaq * plaq_musk  # [batch_size, L, L]
            
            # Add channel dimension before concatenation
            sin_feature = torch.sin(plaq_masked) 
            cos_feature = torch.cos(plaq_masked) 
            features = torch.stack([sin_feature, cos_feature], dim=1)  # [batch_size, 2, L, L]
            
            # Forward pass and accumulate result
            K0 += self.model(features) * field_musk
                
        return K0
            
    def forward(self, theta):
        """
        Transform theta_new to theta_ori
        Input: theta with shape [batch_size, 2, L, L]
        Output: theta with shape [batch_size, 2, L, L]
        """

        plaq = plaq_from_field_batch(theta)
        plaq_stack = torch.stack([plaq, plaq], dim=1)  # [batch_size, 2, L, L]
        K0 = self.compute_K0(theta)
        
        return theta + K0 * plaq_stack

    def inverse(self, theta):
        """
        Transform theta_ori to theta_new
        Input: theta with shape [batch_size, 2, L, L]
        Output: theta with shape [batch_size, 2, L, L]
        """

        plaq = plaq_from_field_batch(theta)
        plaq_stack = torch.stack([plaq, plaq], dim=1)  # [batch_size, 2, L, L]
        K0 = self.compute_K0(theta)
        
        return theta - K0 * plaq_stack

    def field_transformation(self, theta):
        """
        Field transformation function for HMC.
        Input: theta with shape [2, L, L]
        Output: theta with shape [2, L, L]
        """
        
        # Add batch dimension for single input
        theta_batch = theta.unsqueeze(0)
        theta_transformed = self.forward(theta_batch)
        
        # Remove batch dimension
        return theta_transformed.squeeze(0) 
    
    def compute_jac_logdet(self, theta):
        """
        Compute the log determinant of the Jacobian of the field transformation.
        
        theta: [batch_size, 2, L, L]
        """
        batch_size = theta.shape[0]
        K0 = self.compute_K0(theta)
        # jac_det = torch.prod(K0 + 1, dim=(1, 2, 3))
        # Chain the product operations over multiple dimensions
        jac_det = (K0 + 1).prod(dim=1).prod(dim=1).prod(dim=1)
        
        return torch.sum(torch.log(jac_det)) / batch_size
    
    def compute_action(self, theta, beta):
        """
        Compute action for given configuration
        Input: theta with shape [batch_size, 2, L, L]
        """
        batch_size = theta.shape[0]
        plaq = plaq_from_field_batch(theta)
        total_action = torch.sum(torch.cos(plaq))
        
        return -beta * total_action / batch_size  # Average over batch
    
    def compute_force(self, theta, beta, transformed=False):
        """
        Compute force (gradient of action)
        Input: theta with shape [batch_size, 2, L, L]
        """
        theta.requires_grad_(True)
        
        if transformed:
            theta_ori = self.forward(theta)
            action = self.compute_action(theta_ori, beta)
            jac_logdet = self.compute_jac_logdet(theta)
            
            total_action = action - jac_logdet
        else:
            total_action = self.compute_action(theta, beta)
            
        force = torch.autograd.grad(total_action, theta, create_graph=True)[0]
        
        return force.squeeze(0) 
    
    
    def train_step(self, theta_ori, beta):
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
