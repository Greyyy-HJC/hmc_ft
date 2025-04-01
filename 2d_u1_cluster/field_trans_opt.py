# %%
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from utils import plaq_from_field_batch, rect_from_field_batch, get_field_mask, get_plaq_mask, get_rect_mask

# Optimized combined CNN model
class UnifiedCNN(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=32, out_channels_plaq=4, out_channels_rect=8):
        super(UnifiedCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_plaq = nn.Conv2d(hidden_channels, out_channels_plaq, kernel_size=1)
        self.conv_rect = nn.Conv2d(hidden_channels, out_channels_rect, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass with a single input tensor.
        
        Args:
            x: Input tensor with shape [batch_size, channels, height, width]
            
        Returns:
            tuple: (K0, K1) coefficient tensors
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        K0 = self.conv_plaq(x)
        K1 = self.conv_rect(x)
        return K0, K1
        
    def forward_split(self, plaq_features: torch.Tensor, rect_features: torch.Tensor) -> tuple:
        """
        Forward pass with separate plaquette and rectangle features.
        
        Args:
            plaq_features: Plaquette features with shape [batch_size, 2, L, L]
            rect_features: Rectangle features with shape [batch_size, 4, L, L]
            
        Returns:
            tuple: (K0, K1) coefficient tensors
        """
        # Concatenate along channel dimension
        x = torch.cat([plaq_features, rect_features], dim=1)
        return self.forward(x)

# Optimized feature computations without excessive torch.roll
@torch.jit.script
def compute_plaq_features(plaq: torch.Tensor, plaq_mask: torch.Tensor):
    # plaq shape: [batch_size, L, L]
    # Output shape: [batch_size, 2, L, L]
    plaq_masked = plaq * plaq_mask
    return torch.cat([torch.sin(plaq_masked).unsqueeze(1), torch.cos(plaq_masked).unsqueeze(1)], dim=1)

@torch.jit.script
def compute_rect_features(rect: torch.Tensor, rect_mask: torch.Tensor):
    # rect shape: [batch_size, 2, L, L]
    # rect_mask shape: [batch_size, 2, L, L]
    # Output shape: [batch_size, 2, L, L]
    rect_dir0 = rect[:, 0:1] * rect_mask[:, 0:1]  # First direction
    rect_dir1 = rect[:, 1:2] * rect_mask[:, 1:2]  # Second direction
    
    sin_features = torch.cat([torch.sin(rect_dir0), torch.sin(rect_dir1)], dim=1)
    cos_features = torch.cat([torch.cos(rect_dir0), torch.cos(rect_dir1)], dim=1)
    
    return torch.cat([sin_features, cos_features], dim=1)  # Shape: [batch_size, 4, L, L]

# Replace expensive rolls with precomputed indexing
@torch.jit.script
def compute_plaq_phase(plaq: torch.Tensor):
    sin_plaq_dir0_1 = -torch.sin(plaq)
    sin_plaq_dir0_2 = torch.sin(plaq[:, :, torch.arange(-1, plaq.size(2)-1, device=plaq.device)])
    sin_plaq_dir1_1 = torch.sin(plaq)
    sin_plaq_dir1_2 = -torch.sin(plaq[:, torch.arange(-1, plaq.size(1)-1, device=plaq.device), :])
    return sin_plaq_dir0_1, sin_plaq_dir0_2, sin_plaq_dir1_1, sin_plaq_dir1_2

@torch.jit.script
def compute_rect_phase(rect_dir0: torch.Tensor, rect_dir1: torch.Tensor):
    s0, s1 = rect_dir0.shape[1:], rect_dir1.shape[1:]
    idx_roll1 = torch.arange(-1, s0[0]-1, device=rect_dir0.device)
    idx_roll2 = torch.arange(-1, s0[1]-1, device=rect_dir0.device)

    sin_rect_dir0 = [
        -torch.sin(rect_dir0[:, idx_roll1, :]),
        torch.sin(rect_dir0[:, idx_roll1][:, :, idx_roll2]),
        -torch.sin(rect_dir0),
        torch.sin(rect_dir0[:, :, idx_roll2])
    ]

    sin_rect_dir1 = [
        torch.sin(rect_dir1[:, :, idx_roll2]),
        -torch.sin(rect_dir1[:, idx_roll1][:, :, idx_roll2]),
        torch.sin(rect_dir1),
        -torch.sin(rect_dir1[:, idx_roll1, :])
    ]

    return sin_rect_dir0 + sin_rect_dir1

# Updated FieldTransformation class (only key methods shown for brevity)
class FieldTransformation:
    def __init__(self, lattice_size, device='cpu', n_subsets=8, if_check_jac=False, num_workers=0):
        self.L = lattice_size
        self.device = torch.device(device)
        self.n_subsets = n_subsets
        self.if_check_jac = if_check_jac
        self.num_workers = num_workers
        
        # Create models without JIT scripting
        self.models = nn.ModuleList([
            UnifiedCNN(in_channels=6).to(device) for _ in range(n_subsets)
        ])
        
        # Create optimizers and schedulers
        self.optimizers = [torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) for model in self.models]
        self.schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5) for opt in self.optimizers]
        
        # Flag to track if we're in training mode
        self.training = True

    def compute_K0_K1(self, theta, index):
        batch_size = theta.shape[0]
        plaq = plaq_from_field_batch(theta)
        rect = rect_from_field_batch(theta)

        plaq_mask = get_plaq_mask(index, batch_size, self.L).to(self.device)
        rect_mask = get_rect_mask(index, batch_size, self.L).to(self.device)

        plaq_features = compute_plaq_features(plaq, plaq_mask)
        rect_features = compute_rect_features(rect, rect_mask)

        return self.models[index].forward_split(plaq_features, rect_features)

    def ft_phase(self, theta, index):
        batch_size = theta.shape[0]
        plaq = plaq_from_field_batch(theta)
        rect = rect_from_field_batch(theta)

        sin_plaq_stack = torch.stack(compute_plaq_phase(plaq), dim=1)
        K0, K1 = self.compute_K0_K1(theta, index)

        ft_phase_plaq = torch.stack([
            K0[:, 0] * sin_plaq_stack[:, 0] + K0[:, 1] * sin_plaq_stack[:, 1],
            K0[:, 2] * sin_plaq_stack[:, 2] + K0[:, 3] * sin_plaq_stack[:, 3]
        ], dim=1)

        rect_dir0, rect_dir1 = rect[:, 0], rect[:, 1]
        sin_rect_stack = torch.stack(compute_rect_phase(rect_dir0, rect_dir1), dim=1)

        ft_phase_rect = torch.stack([
            torch.sum(K1[:, :4] * sin_rect_stack[:, :4], dim=1),
            torch.sum(K1[:, 4:] * sin_rect_stack[:, 4:], dim=1)
        ], dim=1)

        field_mask = get_field_mask(index, batch_size, self.L).to(self.device)
        return (ft_phase_plaq + ft_phase_rect) * field_mask

    def forward(self, theta):
        """
        Transform theta through all subsets sequentially
        
        Args:
            theta: Field configuration with shape [batch_size, 2, L, L]
            
        Returns:
            Transformed field configuration with shape [batch_size, 2, L, L]
        """
        theta_curr = theta.clone()
        for index in range(self.n_subsets):
            phase = self.ft_phase(theta_curr, index)
            theta_curr = theta_curr + phase
        return theta_curr

    def field_transformation(self, theta):
        """
        Field transformation function for HMC.
        
        Args:
            theta: Input field configuration with shape [2, L, L]
            
        Returns:
            Transformed field configuration with shape [2, L, L]
        """
        # Add batch dimension for single input
        theta_batch = theta.unsqueeze(0).to(self.device)
        
        # Apply forward transformation
        with torch.no_grad():
            theta_transformed = self.forward(theta_batch)
        
        # Remove batch dimension
        return theta_transformed.squeeze(0)

    def inverse(self, theta):
        """
        Transform theta_ori to theta_new sequentially through all subsets
        Uses fixed-point iteration to find the inverse transformation
        """
        theta_curr = theta.clone()
        max_iter = 100
        tol = 1e-6
        
        for index in range(self.n_subsets):
            theta_iter = theta_curr.clone()
            
            # Fixed-point iteration to find inverse transformation for this subset
            for i in range(max_iter):
                # Compute the phase for current iteration
                inv_phase = -self.ft_phase(theta_iter, index)
                
                # Update theta using the inverse phase
                theta_next = theta_curr - inv_phase
                
                # Check convergence
                diff = torch.norm(theta_next - theta_iter) / torch.norm(theta_iter)
                
                if diff < tol:
                    theta_curr = theta_next
                    break
                    
                theta_iter = theta_next
            
            # Warning if not converged
            if diff >= tol:
                print(f"Warning: Inverse iteration for subset {index} did not converge, final diff = {diff:.2e}")
        
        return theta_curr

    def compute_jac_logdet(self, theta):
        """Compute total log determinant of Jacobian for all subsets"""
        try:
            batch_size = theta.shape[0]
            log_det = torch.zeros(batch_size, device=self.device)
            theta_curr = theta.clone()
            
            for index in range(self.n_subsets):
                field_mask = get_field_mask(index, batch_size, self.L).to(self.device)
                
                # Get plaquette and rectangle values
                plaq = plaq_from_field_batch(theta_curr)
                rect = rect_from_field_batch(theta_curr)
                rect_dir0 = rect[:, 0, :, :]  # [batch_size, L, L]
                rect_dir1 = rect[:, 1, :, :]  # [batch_size, L, L]
                
                # For the link in direction 0 or 1, there are two related plaquettes, note there is an extra derivative
                cos_plaq_dir0_1 = -torch.cos(plaq)  # [batch_size, L, L]
                cos_plaq_dir0_2 = -torch.cos(torch.roll(plaq, shifts=1, dims=2))
                
                cos_plaq_dir1_1 = -torch.cos(plaq)
                cos_plaq_dir1_2 = -torch.cos(torch.roll(plaq, shifts=1, dims=1))
                
                cos_plaq_stack = torch.stack([
                    cos_plaq_dir0_1, cos_plaq_dir0_2, 
                    cos_plaq_dir1_1, cos_plaq_dir1_2
                ], dim=1)  # [batch_size, 4, L, L]
                
                # Get K0, K1 coefficients
                K0, K1 = self.compute_K0_K1(theta_curr, index)
                
                # Calculate plaquette Jacobian contribution
                temp = K0 * cos_plaq_stack
                plaq_jac_shift = torch.stack([ 
                    temp[:, 0] + temp[:, 1],  # dir 0 
                    temp[:, 2] + temp[:, 3]   # dir 1
                ], dim=1)  # [batch_size, 2, L, L]
                plaq_jac_shift = plaq_jac_shift * field_mask
                
                # For the link in direction 0 or 1, there are four related rectangles, note there is an extra derivative
                cos_rect_dir0_1 = -torch.cos(torch.roll(rect_dir0, shifts=1, dims=1))  # [batch_size, L, L]
                cos_rect_dir0_2 = -torch.cos(torch.roll(rect_dir0, shifts=(1, 1), dims=(1, 2)))
                cos_rect_dir0_3 = -torch.cos(rect_dir0)
                cos_rect_dir0_4 = -torch.cos(torch.roll(rect_dir0, shifts=1, dims=2))
                
                cos_rect_dir1_1 = -torch.cos(torch.roll(rect_dir1, shifts=1, dims=2))
                cos_rect_dir1_2 = -torch.cos(torch.roll(rect_dir1, shifts=(1, 1), dims=(1, 2)))
                cos_rect_dir1_3 = -torch.cos(rect_dir1)
                cos_rect_dir1_4 = -torch.cos(torch.roll(rect_dir1, shifts=1, dims=1))
                
                cos_rect_stack = torch.stack([
                    cos_rect_dir0_1, cos_rect_dir0_2, cos_rect_dir0_3, cos_rect_dir0_4,
                    cos_rect_dir1_1, cos_rect_dir1_2, cos_rect_dir1_3, cos_rect_dir1_4
                ], dim=1)  # [batch_size, 8, L, L]
                
                # Calculate rectangle Jacobian contribution
                temp = K1 * cos_rect_stack
                rect_jac_shift = torch.stack([
                    temp[:, 0] + temp[:, 1] + temp[:, 2] + temp[:, 3],  # dir 0
                    temp[:, 4] + temp[:, 5] + temp[:, 6] + temp[:, 7]   # dir 1
                ], dim=1)  # [batch_size, 2, L, L]
                rect_jac_shift = rect_jac_shift * field_mask
                
                # Make sure we're not getting log of a non-positive number
                jac_terms = 1 + plaq_jac_shift + rect_jac_shift
                # Add a small epsilon to ensure positivity
                jac_terms = torch.clamp(jac_terms, min=1e-10)
                
                # Accumulate log determinant
                log_det += torch.log(jac_terms).sum(dim=(1, 2, 3))
                
                # Update theta for next subset
                theta_curr = theta_curr + self.ft_phase(theta_curr, index)
            
            return log_det
        except Exception as e:
            print(f"Error in compute_jac_logdet: {str(e)}")
            # Return a zero tensor with gradient
            return torch.zeros(batch_size, device=self.device, requires_grad=True)
    
    def compute_jac_logdet_autograd(self, theta):
        """
        Compute total log determinant of Jacobian using autograd
        This is used for verification purposes
        
        Args:
            theta: Field configuration with shape [batch_size, 2, L, L]
            
        Returns:
            Log determinant of Jacobian
        """
        try:
            # Only take the first sample in batch to reduce computation
            theta_curr = theta.clone()
            theta_curr = theta_curr[0].unsqueeze(0).detach().requires_grad_(True)
            
            # Create function to compute Jacobian for
            def forward_fn(x):
                return self.forward(x)
            
            # Compute Jacobian matrix using autograd
            # We need to set requires_grad=True for all models
            for model in self.models:
                for param in model.parameters():
                    param.requires_grad_(False)
                    
            jac = torch.autograd.functional.jacobian(forward_fn, theta_curr)
            
            # Restore requires_grad state
            for model in self.models:
                for param in model.parameters():
                    param.requires_grad_(True)
                    
            # Reshape to 2D matrix for determinant calculation
            jac_2d = jac.reshape(theta_curr.numel(), theta_curr.numel())
            
            # Compute log determinant
            log_det = torch.logdet(jac_2d)
            
            return log_det.unsqueeze(0)  # Add batch dimension back
        except Exception as e:
            print(f"Error in Jacobian autograd calculation: {str(e)}")
            # Return a dummy value
            return torch.tensor([0.0], device=self.device)
    
    def compute_action(self, theta, beta):
        """
        Compute action for given configuration
        
        Args:
            theta: Field configuration with shape [batch_size, 2, L, L]
            beta: Coupling constant (float)
            
        Returns:
            Action values with shape [batch_size]
        """
        # Calculate plaquettes
        plaq = plaq_from_field_batch(theta)  # [batch_size, L, L]
        
        # Sum cosine of plaquettes over spatial dimensions
        total_action = torch.sum(torch.cos(plaq), dim=(1, 2))
        
        # Apply beta factor
        return -beta * total_action
    
    def compute_force(self, theta, beta, transformed=False):
        """
        Compute force (gradient of action)
        
        Args:
            theta: Field configuration with shape [batch_size, 2, L, L]
            beta: Coupling constant (float)
            transformed: Whether to compute force in transformed space (bool)
            
        Returns:
            Force with shape [batch_size, 2, L, L]
        """
        theta = theta.clone().detach().requires_grad_(True)
        
        if transformed:
            # In transformed space, we need to account for the Jacobian
            theta_ori = self.forward(theta)
            action = self.compute_action(theta_ori, beta)
            jac_logdet = self.compute_jac_logdet(theta)
            
            # Verify Jacobian calculation if requested
            if self.if_check_jac:
                try:
                    jac_logdet_autograd = self.compute_jac_logdet_autograd(theta)
                    diff = (jac_logdet_autograd[0] - jac_logdet[0]) / jac_logdet[0]
                    
                    if abs(diff.item()) > 1e-4:
                        print(f"Jacobian log determinant difference = {diff:.2f}")
                        print("Jacobian is not correct!")
                    else:
                        print(f"Jacobian log determinant by hand is {jac_logdet[0]:.2e}")
                        print(f"Jacobian log determinant by autograd is {jac_logdet_autograd[0]:.2e}")
                        print("Jacobian is all good")
                except Exception as e:
                    print(f"Error computing Jacobian: {str(e)}")
            
            # Total action includes the Jacobian contribution
            total_action = action - jac_logdet
        else:
            # In original space, just compute the action
            total_action = self.compute_action(theta, beta)
        
        # Calculate force using autograd
        try:
            # Sum over batch dimension to get scalar for gradient calculation
            force = torch.autograd.grad(total_action.sum(), theta, create_graph=self.training)[0]
            return force  # Keep gradients if needed for backward pass
        except Exception as e:
            print(f"Error in gradient calculation: {str(e)}")
            # Return zero force as fallback
            return torch.zeros_like(theta)
    
    def loss_fn(self, theta_ori):
        """
        Compute loss function for given configuration
        
        Args:
            theta_ori: Original field configuration with shape [batch_size, 2, L, L]
            
        Returns:
            Loss value (scalar)
        """
        # Ensure theta_ori doesn't have gradients from previous computation
        theta_ori = theta_ori.detach().requires_grad_(True)
        
        # Transform original configuration to new configuration
        theta_new = self.inverse(theta_ori)
        
        try:
            # Compute forces in original and transformed spaces
            force_ori = self.compute_force(theta_ori, beta=1)
            force_new = self.compute_force(theta_new, self.train_beta, transformed=True)
            
            # Compute loss using multiple norms to ensure good matching across different scales
            vol = self.L * self.L
            loss = torch.norm(force_new - force_ori, p=2) / (vol**(1/2)) + \
                torch.norm(force_new - force_ori, p=4) / (vol**(1/4)) + \
                torch.norm(force_new - force_ori, p=6) / (vol**(1/6)) + \
                torch.norm(force_new - force_ori, p=8) / (vol**(1/8))
            
            return loss
        except Exception as e:
            print(f"Error in loss calculation: {str(e)}")
            # Create a dummy loss that requires grad
            dummy = torch.zeros(1, device=self.device, requires_grad=True)
            return dummy

    def train_step(self, theta_ori):
        """
        Perform a single training step for all subsets together
        
        Args:
            theta_ori: Original field configuration
            
        Returns:
            float: Loss value for this step
        """
        # Set training mode explicitly
        self._set_models_mode(True)
        
        theta_ori = theta_ori.to(self.device)
        
        # Zero all gradients
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        
        try:
            # Compute loss with gradient tracking
            loss = self.loss_fn(theta_ori)
            
            # Backpropagate if loss is valid
            if loss.requires_grad and not torch.isnan(loss).any() and not torch.isinf(loss).any():
                loss.backward()
                
                # Update all models
                for optimizer in self.optimizers:
                    optimizer.step()
                
                return loss.item()
            else:
                print("Warning: Loss doesn't require gradients or contains NaN/Inf values")
                return float('nan')
        except Exception as e:
            print(f"Error during training step: {str(e)}")
            import traceback
            traceback.print_exc()
            return float('nan')

    def evaluate_step(self, theta_ori):
        """
        Perform a single evaluation step
        
        Args:
            theta_ori: Original field configuration
            
        Returns:
            float: Loss value for this evaluation step
        """
        # Set evaluation mode explicitly
        self._set_models_mode(False)
        
        theta_ori = theta_ori.to(self.device)
        
        # Compute loss with gradient tracking off
        with torch.no_grad():
            try:
                loss = self.loss_fn(theta_ori)
                # Check if loss is valid
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    return float('nan')
                return loss.item()
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                return float('nan')

    def train(self, train_data, test_data, train_beta, n_epochs=100, batch_size=4):
        """
        Train all models together
        
        Args:
            train_data: Training dataset
            test_data: Testing dataset
            train_beta: Beta value for training
            n_epochs: Number of training epochs
            batch_size: Batch size for training
        """
        train_losses = []
        test_losses = []
        best_loss = float('inf')
        
        self.train_beta = train_beta
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, num_workers=self.num_workers, pin_memory=False
        )
        
        try:
            for epoch in tqdm(range(n_epochs), desc="Training epochs"):
                print(f"\nEpoch {epoch+1}/{n_epochs}")
                
                # Training phase
                epoch_losses = []
                
                for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)):
                    loss = self.train_step(batch)
                    if not np.isnan(loss):
                        epoch_losses.append(loss)
                    
                    # Log every 10 batches
                    if (i+1) % 10 == 0 and len(epoch_losses) > 0:
                        print(f"  Batch {i+1}: Current loss = {epoch_losses[-1]:.6f}, Avg loss = {np.mean(epoch_losses):.6f}")
                
                if len(epoch_losses) > 0:
                    train_loss = np.mean(epoch_losses)
                    train_losses.append(train_loss)
                    print(f"Training complete. Average loss: {train_loss:.6f}")
                else:
                    print("Warning: No valid losses in this epoch")
                    train_loss = float('nan')
                    train_losses.append(train_loss)
                
                # Evaluation phase
                print("Evaluating model on test data...")
                test_losses_epoch = []
                
                for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                    loss = self.evaluate_step(batch)
                    if not np.isnan(loss):
                        test_losses_epoch.append(loss)
                
                if len(test_losses_epoch) > 0:
                    test_loss = np.mean(test_losses_epoch)
                    test_losses.append(test_loss)
                    print(f"Evaluation complete. Test loss: {test_loss:.6f}")
                else:
                    print("Warning: No valid test losses in this epoch")
                    test_loss = float('nan')
                    test_losses.append(test_loss)
                
                # Print epoch summary
                print(f"Epoch {epoch+1}/{n_epochs} - "
                    f"Train Loss: {train_loss:.6f} - "
                    f"Test Loss: {test_loss:.6f}")
                
                # Save best model if test loss improved
                if test_loss < best_loss and not np.isnan(test_loss):
                    self._save_best_model(epoch, test_loss)
                    best_loss = test_loss
                    print(f"New best test loss: {test_loss:.6f}")
                
                # Update learning rate schedulers
                if not np.isnan(test_loss):
                    old_lrs = [param_group['lr'] for optimizer in self.optimizers for param_group in optimizer.param_groups]
                    self._update_schedulers(test_loss)
                    new_lrs = [param_group['lr'] for optimizer in self.optimizers for param_group in optimizer.param_groups]
                    
                    if old_lrs != new_lrs:
                        print(f"Learning rates updated: {old_lrs} -> {new_lrs}")
            
            # Plot training history
            if len(train_losses) > 0 and len(test_losses) > 0:
                self._plot_training_history(train_losses, test_losses)
            
            # Load best model
            self._load_best_model(train_beta)
            
        except KeyboardInterrupt:
            print("Training interrupted by user")
        except Exception as e:
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _set_models_mode(self, is_train):
        """
        Set all models to training or evaluation mode
        
        Args:
            is_train: If True, set to training mode, otherwise evaluation mode
        """
        self.training = is_train
        mode_func = lambda model: model.train() if is_train else model.eval()
        
        for model in self.models:
            mode_func(model)
    
    def _update_schedulers(self, test_loss):
        """
        Update all learning rate schedulers
        
        Args:
            test_loss: Current test loss value
        """
        for scheduler in self.schedulers:
            scheduler.step(test_loss)
    
    def _save_best_model(self, epoch, loss):
        """
        Save the best model checkpoint
        
        Args:
            epoch: Current epoch number
            loss: Current loss value
        """
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        save_dict = {
            'epoch': epoch,
            'loss': loss,
        }
        # Save state dict for each model
        for i, model in enumerate(self.models):
            save_dict[f'model_state_dict_{i}'] = model.state_dict()
        for i, optimizer in enumerate(self.optimizers):
            save_dict[f'optimizer_state_dict_{i}'] = optimizer.state_dict()
        torch.save(save_dict, f'models/best_model_L{self.L}_train_beta{self.train_beta}.pt')
        print(f"Saved best model checkpoint to models/best_model_L{self.L}_train_beta{self.train_beta}.pt")

    def _plot_training_history(self, train_losses, test_losses):
        """
        Plot and save training history
        
        Args:
            train_losses: List of training losses
            test_losses: List of testing losses
        """
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train')
        plt.plot(test_losses, label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots/cnn_loss_L{self.L}_train_beta{self.train_beta}.pdf', transparent=True)
        plt.show()

    def _load_best_model(self, train_beta):
        """
        Load the best model from checkpoint for all subsets
        
        Args:
            train_beta: Beta value used during training
        """
        checkpoint_path = f'models/best_model_L{self.L}_train_beta{train_beta}.pt'
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file {checkpoint_path} does not exist. Skipping model loading.")
            return
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load models
            for i, model in enumerate(self.models):
                if f'model_state_dict_{i}' in checkpoint:
                    model.load_state_dict(checkpoint[f'model_state_dict_{i}'])
                else:
                    print(f"Warning: Could not find state dict for model {i} in checkpoint")
            
            print(f"Loaded best models from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            print("Continuing with randomly initialized model")