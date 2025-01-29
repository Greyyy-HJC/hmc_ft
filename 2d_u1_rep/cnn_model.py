# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import plaq_from_field_batch

def get_mask(index, batch_size, L):
    '''
    Get mask indices for a configuration with shape [batch_size, 2, L, L]
    Get mask indices for plaquette phase angles with shape [batch_size, L, L]
    '''
    
    field_mask = torch.zeros((batch_size, 2, L, L), dtype=torch.bool)
    plaq_mask = torch.zeros((batch_size, L, L), dtype=torch.bool)
    
    if index == 0:
        field_mask[:, 0, 0::2, 0::2] = True
        plaq_mask[:, 1::2, :] = True
        
    elif index == 1:
        field_mask[:, 0, 0::2, 1::2] = True
        plaq_mask[:, 1::2, :] = True
        
    elif index == 2:
        field_mask[:, 0, 1::2, 0::2] = True
        plaq_mask[:, 0::2, :] = True

    elif index == 3:
        field_mask[:, 0, 1::2, 1::2] = True
        plaq_mask[:, 0::2, :] = True
        
    elif index == 4:
        field_mask[:, 1, 0::2, 0::2] = True
        plaq_mask[:, :, 1::2] = True
        
    elif index == 5:
        field_mask[:, 1, 0::2, 1::2] = True
        plaq_mask[:, :, 0::2] = True
        
    elif index == 6:
        field_mask[:, 1, 1::2, 0::2] = True
        plaq_mask[:, :, 1::2] = True

    elif index == 7:
        field_mask[:, 1, 1::2, 1::2] = True
        plaq_mask[:, :, 0::2] = True

    return field_mask, plaq_mask
    
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
        
    def compute_K0(self, theta, index):
        """
        Compute K0 for given theta
        Input: theta with shape [batch_size, 2, L, L]
        Output: K0 with shape [batch_size, 2, L, L]
        """
        batch_size = theta.shape[0]
        K0 = torch.zeros((batch_size, 2, self.L, self.L), device=self.device)
        
        # Compute plaquettes for all batches at once
        plaq = plaq_from_field_batch(theta)  # [batch_size, L, L]
        
        field_mask, plaq_mask = get_mask(index, batch_size, self.L)
        field_mask = field_mask.to(self.device)
        plaq_mask = plaq_mask.to(self.device)
        
        # Apply mask and compute features for all batches at once
        plaq_masked = plaq * plaq_mask  # [batch_size, L, L]
        
        # Add channel dimension before concatenation
        sin_feature = torch.sin(plaq_masked) 
        cos_feature = torch.cos(plaq_masked) 
        features = torch.stack([sin_feature, cos_feature], dim=1)  # [batch_size, 2, L, L]
        
        # Forward pass and accumulate result
        K0 += self.model(features) * field_mask
                
        return K0
    
    def ft_phase(self, theta):
        """
        Compute the phase factor for field transformation
        Input: theta with shape [batch_size, 2, L, L]
        Output: phase with shape [batch_size, 2, L, L]
        """
        plaq = plaq_from_field_batch(theta)
        sin_plaq = torch.sin(plaq)
        sin_plaq_stack = torch.stack([sin_plaq, -sin_plaq], dim=1)  # [batch_size, 2, L, L] 
        K0 = self.compute_K0(theta, index=4)
        return K0 * sin_plaq_stack
            
    def forward(self, theta):
        """
        Transform theta_new to theta_ori
        Input: theta with shape [batch_size, 2, L, L]
        Output: theta with shape [batch_size, 2, L, L]
        """
        
        return theta - self.ft_phase(theta)

    def inverse(self, theta):
        """
        Transform theta_ori to theta_new
        Input: theta with shape [batch_size, 2, L, L]
        Output: theta with shape [batch_size, 2, L, L]
        """
        
        theta_curr = theta
        max_iter = 100
        tol = 1e-6
        
        for i in range(max_iter):
            inv_phase = - self.ft_phase(theta_curr)
            theta_next = theta - inv_phase
            
            # calculate relative error
            diff = torch.norm(theta_next - theta_curr) / torch.norm(theta_curr)
            
            if diff < tol:
                return theta_next
                    
            theta_curr = theta_next
            
        print(f"Warning: Inverse iteration did not converge, final diff = {diff:.2e}")
        return theta_curr

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
        Output: log_det with shape [batch_size]
        """
        K0 = self.compute_K0(theta, index=4)
        plaq = plaq_from_field_batch(theta)
        cos_plaq_stack = torch.stack([torch.cos(plaq), torch.cos(plaq)], dim=1) # [batch_size, 2, L, L]
        log_det = torch.log(1 - K0 * cos_plaq_stack).sum(dim=1).sum(dim=1).sum(dim=1)

        return log_det
    
    def compute_action(self, theta, beta):
        """
        Compute action for given configuration
        Input: theta with shape [batch_size, 2, L, L]; beta is a float
        Output: action with shape [batch_size]
        """
        plaq = plaq_from_field_batch(theta) # [batch_size, L, L]
        total_action = torch.sum(torch.cos(plaq), dim=1).sum(dim=1)
        
        return -beta * total_action
    
    def compute_force(self, theta, beta, transformed=False):
        """
        Compute force (gradient of action)
        Input: theta with shape [batch_size, 2, L, L]; beta is a float
        Output: force with shape [batch_size, 2, L, L]
        """
        batch_size = theta.shape[0]
        
        if transformed:
            theta_ori = self.forward(theta)
            action = self.compute_action(theta_ori, beta)
            jac_logdet = self.compute_jac_logdet(theta)
            total_action = action - jac_logdet
        else:
            total_action = self.compute_action(theta, beta)
        
        # calculate force for each sample in batch
        force = torch.zeros_like(theta)
        for i in range(batch_size):
            grad = torch.autograd.grad(total_action[i], theta, create_graph=True)[0]
            force[i] = grad[i] 
        
        return force  # shape: [batch_size, 2, L, L]

    def train_step(self, theta_ori, beta):
        """Single training step
        Args:
            theta_ori (torch.Tensor): Original field configuration
            beta (float): Coupling constant
        Returns:
            float: Loss value
        """
        theta_ori = theta_ori.to(self.device)
        
        with torch.autograd.set_grad_enabled(True):
            theta_new = self.inverse(theta_ori)
            force_ori = self.compute_force(theta_new, beta=2.5)
            force_new = self.compute_force(theta_new, beta, transformed=True)
            
            vol = self.L * self.L
            loss = torch.norm(force_new - force_ori, p=2) / (vol**(1/2))
            
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
        return loss.item()

    def evaluate_step(self, theta_ori, beta):
        """Single evaluation step
        Args:
            theta_ori (torch.Tensor): Original field configuration
            beta (float): Coupling constant
        Returns:
            float: Loss value
        """
        theta_ori = theta_ori.to(self.device)
        
        theta_new = self.inverse(theta_ori)
        force_ori = self.compute_force(theta_new, beta=2.5)
        force_new = self.compute_force(theta_new, beta, transformed=True)
        
        vol = self.L * self.L
        loss = torch.norm(force_new - force_ori, p=2) / (vol**(1/2))
            
        return loss.item()

    def train(self, train_data, test_data, beta, n_epochs=100, batch_size=4, patience=10):
        """Train the model with early stopping"""
        train_losses = []
        test_losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size
        )
        
        for epoch in tqdm(range(n_epochs), desc="Training epochs"):
            # Training phase
            self.model.train()
            epoch_losses = []
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
                loss = self.train_step(batch, beta)
                epoch_losses.append(loss)
                
            train_loss = np.mean(epoch_losses)
            train_losses.append(train_loss)
            
            # Evaluation phase
            self.model.eval()
            test_losses_epoch = []
            
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                loss = self.evaluate_step(batch, beta)
                test_losses_epoch.append(loss)
                
            test_loss = np.mean(test_losses_epoch)
            test_losses.append(test_loss)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{n_epochs} - "
                  f"Train Loss: {train_loss:.6f} - "
                  f"Test Loss: {test_loss:.6f}")
            
            # Early stopping and model saving
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_loss,
                }, 'models/best_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            self.scheduler.step(test_loss)
        
        # Plot training history
        self._plot_training_history(train_losses, test_losses)
        
        # Load best model
        self._load_best_model()

    def _plot_training_history(self, train_losses, test_losses):
        """Plot and save training history"""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train')
        plt.plot(test_losses, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/cnn_loss.pdf', transparent=True)
        plt.show()

    def _load_best_model(self):
        """Load the best model from checkpoint"""
        checkpoint = torch.load('models/best_model.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}") 
