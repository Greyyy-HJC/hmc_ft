# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.autograd.functional as F

from utils import plaq_from_field_batch, rect_from_field_batch, get_field_mask, get_plaq_mask, get_rect_mask
from cnn_model_jit import plaqCNN, rectCNN, combineCNN

# Add TorchScript optimized functions
@torch.jit.script
def compute_plaq_features(plaq: torch.Tensor, plaq_mask: torch.Tensor) -> torch.Tensor:
    plaq_masked = plaq * plaq_mask
    plaq_sin_feature = torch.sin(plaq_masked)
    plaq_cos_feature = torch.cos(plaq_masked)
    return torch.stack([plaq_sin_feature, plaq_cos_feature], dim=1)

@torch.jit.script
def compute_rect_features(rect: torch.Tensor, rect_mask: torch.Tensor) -> torch.Tensor:
    rect_masked = rect * rect_mask
    rect_sin_feature = torch.sin(rect_masked)
    rect_cos_feature = torch.cos(rect_masked)
    return torch.cat([rect_sin_feature, rect_cos_feature], dim=1)

@torch.jit.script
def compute_plaq_phase(plaq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sin_plaq_dir0_1 = -torch.sin(plaq)
    sin_plaq_dir0_2 = torch.sin(torch.roll(plaq, shifts=1, dims=2))
    sin_plaq_dir1_1 = torch.sin(plaq)
    sin_plaq_dir1_2 = -torch.sin(torch.roll(plaq, shifts=1, dims=1))
    return sin_plaq_dir0_1, sin_plaq_dir0_2, sin_plaq_dir1_1, sin_plaq_dir1_2

@torch.jit.script
def compute_rect_phase(rect_dir0: torch.Tensor, rect_dir1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sin_rect_dir0_1 = -torch.sin(torch.roll(rect_dir0, shifts=1, dims=1))
    sin_rect_dir0_2 = torch.sin(torch.roll(rect_dir0, shifts=(1, 1), dims=(1, 2)))
    sin_rect_dir0_3 = -torch.sin(rect_dir0)
    sin_rect_dir0_4 = torch.sin(torch.roll(rect_dir0, shifts=1, dims=2))
    
    sin_rect_dir1_1 = torch.sin(torch.roll(rect_dir1, shifts=1, dims=2))
    sin_rect_dir1_2 = -torch.sin(torch.roll(rect_dir1, shifts=(1, 1), dims=(1, 2)))
    sin_rect_dir1_3 = torch.sin(rect_dir1)
    sin_rect_dir1_4 = -torch.sin(torch.roll(rect_dir1, shifts=1, dims=1))
    
    return sin_rect_dir0_1, sin_rect_dir0_2, sin_rect_dir0_3, sin_rect_dir0_4, sin_rect_dir1_1, sin_rect_dir1_2, sin_rect_dir1_3, sin_rect_dir1_4

class FieldTransformation:
    """Neural network based field transformation"""
    def __init__(self, lattice_size, device='cpu', n_subsets=8, if_check_jac=False, use_combined_model=True, num_workers=0):
        self.L = lattice_size
        self.device = torch.device(device)
        self.n_subsets = n_subsets
        self.if_check_jac = if_check_jac
        self.use_combined_model = use_combined_model
        self.num_workers = num_workers
        
        # Create n_subsets independent models for each subset
        if use_combined_model:
            # Use combined model
            # self.plaq_models = nn.ModuleList([plaqCNN().to(device) for _ in range(n_subsets)])
            # self.rect_models = nn.ModuleList([rectCNN().to(device) for _ in range(n_subsets)])
            # self.combine_models = nn.ModuleList([combineCNN().to(device) for _ in range(n_subsets)])
            
            self.plaq_models = nn.ModuleList([
                torch.jit.script(plaqCNN().to(device)) for _ in range(n_subsets)
            ])
            self.rect_models = nn.ModuleList([
                torch.jit.script(rectCNN().to(device)) for _ in range(n_subsets)
            ])
            self.combine_models = nn.ModuleList([
                torch.jit.script(combineCNN().to(device)) for _ in range(n_subsets)
            ])
            
            self.plaq_optimizers = [
                torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
                for model in self.plaq_models
            ]
            self.rect_optimizers = [
                torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
                for model in self.rect_models
            ]
            self.combine_optimizers = [
                torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
                for model in self.combine_models
            ]
            
            self.plaq_schedulers = [
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5
                )
                for optimizer in self.plaq_optimizers
            ]
            self.rect_schedulers = [
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5
                )
                for optimizer in self.rect_optimizers
            ]
            self.combine_schedulers = [
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5
                )
                for optimizer in self.combine_optimizers
            ]
        else:
            # Use original independent models
            self.plaq_models = nn.ModuleList([plaqCNN().to(device) for _ in range(n_subsets)])
            self.rect_models = nn.ModuleList([rectCNN().to(device) for _ in range(n_subsets)])
            self.plaq_optimizers = [
                torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
                for model in self.plaq_models
            ]
            self.rect_optimizers = [
                torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
                for model in self.rect_models
            ]
            self.plaq_schedulers = [
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5
                )
                for optimizer in self.plaq_optimizers
            ]
            self.rect_schedulers = [
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5
                )
                for optimizer in self.rect_optimizers
            ]

    def compute_K0_K1(self, theta, index):
        """
        Compute K0 and K1 using the combined model
        Input: theta with shape [batch_size, 2, L, L]
        Output: K0 with shape [batch_size, 4, L, L], K1 with shape [batch_size, 8, L, L]
        """
        batch_size = theta.shape[0]
        
        # Calculate plaq features
        plaq = plaq_from_field_batch(theta)
        plaq_mask = get_plaq_mask(index, batch_size, self.L).to(self.device)
        plaq_features = compute_plaq_features(plaq, plaq_mask)
        
        # Calculate rect features
        rect = rect_from_field_batch(theta)
        rect_mask = get_rect_mask(index, batch_size, self.L).to(self.device)
        rect_features = compute_rect_features(rect, rect_mask)
        
        # Use plaqCNN and rectCNN to extract features
        plaq_intermediate = self.plaq_models[index](plaq_features)
        rect_intermediate = self.rect_models[index](rect_features)
        
        # Use combineCNN to generate final K0 and K1
        K0, K1 = self.combine_models[index](plaq_intermediate, rect_intermediate)
        
        return K0, K1

    def compute_K0(self, theta, index):
        """
        Compute K0 for given theta and subset index
        Input: theta with shape [batch_size, 2, L, L]
        Output: K0 with shape [batch_size, 4, L, L]
        """
        if self.use_combined_model:
            K0, _ = self.compute_K0_K1(theta, index)
            return K0
        
        # Original implementation
        batch_size = theta.shape[0]
        K0 = torch.zeros((batch_size, 4, self.L, self.L), device=self.device) 
        
        plaq = plaq_from_field_batch(theta)
        plaq_mask = get_plaq_mask(index, batch_size, self.L).to(self.device)
        plaq_features = compute_plaq_features(plaq, plaq_mask)
        
        # Use the corresponding model for this subset
        K0 += self.plaq_models[index](plaq_features)
                
        return K0
    
    def compute_K1(self, theta, index):
        """
        Compute K1 for given theta and subset index
        Input: theta with shape [batch_size, 4, L, L]
        Output: K1 with shape [batch_size, 4, L, L]
        """
        if self.use_combined_model:
            _, K1 = self.compute_K0_K1(theta, index)
            return K1
        
        # Original implementation
        batch_size = theta.shape[0]
        K1 = torch.zeros((batch_size, 8, self.L, self.L), device=self.device)
        
        rect = rect_from_field_batch(theta)
        rect_mask = get_rect_mask(index, batch_size, self.L).to(self.device)
        rect_features = compute_rect_features(rect, rect_mask)
        
        # Use the corresponding model for this subset
        K1 += self.rect_models[index](rect_features)
                
        return K1

    def ft_phase(self, theta, index):
        """
        Compute the phase factor for field transformation for a specific subset
        """
        batch_size = theta.shape[0]
        plaq = plaq_from_field_batch(theta) # [batch_size, L, L]
        
        # For the link in direction 0 or 1, there are two related plaquettes, note the group derivative is -sin
        sin_plaq_dir0_1, sin_plaq_dir0_2, sin_plaq_dir1_1, sin_plaq_dir1_2 = compute_plaq_phase(plaq)
        
        sin_plaq_stack = torch.stack([sin_plaq_dir0_1, sin_plaq_dir0_2, sin_plaq_dir1_1, sin_plaq_dir1_2], dim=1) # [batch_size, 4, L, L]
        
        K0 = self.compute_K0(theta, index) # [batch_size, 4, L, L]
        
        # Calculate plaquette phase contribution
        temp = K0 * sin_plaq_stack
        ft_phase_plaq = torch.stack([ 
            temp[:, 0] + temp[:, 1], # dir 0  
            temp[:, 2] + temp[:, 3]  # dir 1
        ], dim=1)  # [batch_size, 2, L, L]
        
        # Calculate rectangle phase contribution
        rect = rect_from_field_batch(theta) # [batch_size, 2, L, L]
        rect_dir0 = rect[:, 0, :, :] # [batch_size, L, L]
        rect_dir1 = rect[:, 1, :, :] # [batch_size, L, L]
        
        # For the link in direction 0 or 1, there are four related rectangles, note the group derivative is -sin
        sin_rect_dir0_1, sin_rect_dir0_2, sin_rect_dir0_3, sin_rect_dir0_4, sin_rect_dir1_1, sin_rect_dir1_2, sin_rect_dir1_3, sin_rect_dir1_4 = compute_rect_phase(rect_dir0, rect_dir1)
        
        sin_rect_stack = torch.stack([
            sin_rect_dir0_1, sin_rect_dir0_2, sin_rect_dir0_3, sin_rect_dir0_4, 
            sin_rect_dir1_1, sin_rect_dir1_2, sin_rect_dir1_3, sin_rect_dir1_4
        ], dim=1) # [batch_size, 8, L, L]
        
        K1 = self.compute_K1(theta, index) # [batch_size, 8, L, L]
        temp = K1 * sin_rect_stack
        ft_phase_rect = torch.stack([
            temp[:, 0] + temp[:, 1] + temp[:, 2] + temp[:, 3], # dir 0
            temp[:, 4] + temp[:, 5] + temp[:, 6] + temp[:, 7]  # dir 1
        ], dim=1)  # [batch_size, 2, L, L]
        
        # Apply field mask to the combined phase
        field_mask = get_field_mask(index, batch_size, self.L).to(self.device)
        
        return (ft_phase_plaq + ft_phase_rect) * field_mask

    def forward(self, theta):
        """
        Transform theta_new to theta_ori sequentially through all subsets
        
        Args:
            theta: Input field configuration with shape [batch_size, 2, L, L]
            
        Returns:
            Transformed field configuration with shape [batch_size, 2, L, L]
        """
        theta_curr = theta.clone()
        
        # Apply transformation for each subset sequentially
        for index in range(self.n_subsets):
            theta_curr = theta_curr + self.ft_phase(theta_curr, index)
            
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
        theta_batch = theta.unsqueeze(0)
        
        # Apply forward transformation
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
            
            # Get K0 coefficients
            K0 = self.compute_K0(theta_curr, index)  # [batch_size, 4, L, L]
            
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
            
            # Get K1 coefficients
            K1 = self.compute_K1(theta_curr, index)  # [batch_size, 8, L, L]
            
            # Calculate rectangle Jacobian contribution
            temp = K1 * cos_rect_stack
            rect_jac_shift = torch.stack([
                temp[:, 0] + temp[:, 1] + temp[:, 2] + temp[:, 3],  # dir 0
                temp[:, 4] + temp[:, 5] + temp[:, 6] + temp[:, 7]   # dir 1
            ], dim=1)  # [batch_size, 2, L, L]
            rect_jac_shift = rect_jac_shift * field_mask
            
            # Accumulate log determinant
            log_det += torch.log(1 + plaq_jac_shift + rect_jac_shift).sum(dim=(1, 2, 3))
            
            # Update theta for next subset
            theta_curr = theta_curr + self.ft_phase(theta_curr, index)
        
        return log_det
    
    def compute_jac_logdet_autograd(self, theta):
        """
        Compute total log determinant of Jacobian using autograd
        This is used for verification purposes
        
        Args:
            theta: Field configuration with shape [batch_size, 2, L, L]
            
        Returns:
            Log determinant of Jacobian
        """
        # Only take the first sample in batch to reduce computation
        theta_curr = theta.clone()
        theta_curr = theta_curr[0].unsqueeze(0)
        
        # Compute Jacobian matrix using autograd
        jac = F.jacobian(self.forward, theta_curr)
        
        # Reshape to 2D matrix for determinant calculation
        jac_2d = jac.reshape(theta_curr.shape[0], theta_curr.numel(), theta_curr.numel())
        
        # Compute log determinant
        log_det = torch.logdet(jac_2d)

        return log_det
    
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
        batch_size = theta.shape[0]
        
        if transformed:
            # In transformed space, we need to account for the Jacobian
            theta_ori = self.forward(theta)
            action = self.compute_action(theta_ori, beta)
            jac_logdet = self.compute_jac_logdet(theta)
            
            # Verify Jacobian calculation if requested
            if self.if_check_jac:
                jac_logdet_autograd = self.compute_jac_logdet_autograd(theta)
                
                diff = (jac_logdet_autograd[0] - jac_logdet[0]) / jac_logdet[0]
                
                if abs(diff.item()) > 1e-4:
                    print(f"Jacobian log determinant difference = {diff:.2f}")
                    print("Jacobian is not correct!")
                else:
                    print(f"Jacobian log determinant by hand is {jac_logdet[0]:.2e}")
                    print(f"Jacobian log determinant by autograd is {jac_logdet_autograd[0]:.2e}")
                    print("Jacobian is all good")
            
            # Total action includes the Jacobian contribution
            total_action = action - jac_logdet
        else:
            # In original space, just compute the action
            total_action = self.compute_action(theta, beta)
        
        # Calculate force for each sample in batch
        force = torch.zeros_like(theta)
        for i in range(batch_size):
            grad = torch.autograd.grad(total_action[i], theta, create_graph=True)[0]
            force[i] = grad[i] 
        
        return force  # shape: [batch_size, 2, L, L]
    
    def loss_fn(self, theta_ori):
        """
        Compute loss function for given configuration
        
        Args:
            theta_ori: Original field configuration with shape [batch_size, 2, L, L]
            
        Returns:
            Loss value (scalar)
        """
        # Transform original configuration to new configuration
        theta_new = self.inverse(theta_ori)
        
        # Compute forces in original and transformed spaces
        force_ori = self.compute_force(theta_new, beta=1)
        force_new = self.compute_force(theta_new, self.train_beta, transformed=True)
        
        # Compute loss using multiple norms to ensure good matching across different scales
        vol = self.L * self.L
        loss = torch.norm(force_new - force_ori, p=2) / (vol**(1/2)) + \
               torch.norm(force_new - force_ori, p=4) / (vol**(1/4)) + \
               torch.norm(force_new - force_ori, p=6) / (vol**(1/6)) + \
               torch.norm(force_new - force_ori, p=8) / (vol**(1/8))
        
        return loss

    def train_step(self, theta_ori):
        """
        Perform a single training step for all subsets together
        
        Args:
            theta_ori: Original field configuration
            
        Returns:
            float: Loss value for this step
        """
        theta_ori = theta_ori.to(self.device)
        
        with torch.autograd.set_grad_enabled(True):
            # Compute loss
            loss = self.loss_fn(theta_ori)
            
            # Zero all gradients
            self._zero_all_grads()
            
            # Backpropagate
            loss.backward()
            
            # Update all models
            self._step_all_optimizers()
            
        return loss.item()
    
    def _zero_all_grads(self):
        """Zero gradients for all optimizers"""
        for optimizer in self.plaq_optimizers:
            optimizer.zero_grad()
        for optimizer in self.rect_optimizers:
            optimizer.zero_grad()
        if self.use_combined_model:
            for optimizer in self.combine_optimizers:
                optimizer.zero_grad()
    
    def _step_all_optimizers(self):
        """Step all optimizers"""
        for optimizer in self.plaq_optimizers:
            optimizer.step()
        for optimizer in self.rect_optimizers:
            optimizer.step()
        if self.use_combined_model:
            for optimizer in self.combine_optimizers:
                optimizer.step()

    def evaluate_step(self, theta_ori):
        """
        Perform a single evaluation step
        
        Args:
            theta_ori: Original field configuration
            
        Returns:
            float: Loss value for this evaluation step
        """
        theta_ori = theta_ori.to(self.device)
        
        # * gradient is needed for evaluation
        theta_ori.requires_grad_(True)
        loss = self.loss_fn(theta_ori)
        
        return loss.item()

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
            train_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, num_workers=self.num_workers
        )
        
        for epoch in tqdm(range(n_epochs), desc="Training epochs"):
            # Training phase
            self._set_models_mode(True)  # Set models to training mode
            
            epoch_losses = []
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                
            train_loss = np.mean(epoch_losses)
            train_losses.append(train_loss)
            
            # Evaluation phase
            self._set_models_mode(False)  # Set models to evaluation mode
            
            test_losses_epoch = []
            
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                loss = self.evaluate_step(batch)
                test_losses_epoch.append(loss)
                
            test_loss = np.mean(test_losses_epoch)
            test_losses.append(test_loss)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{n_epochs} - "
                  f"Train Loss: {train_loss:.6f} - "
                  f"Test Loss: {test_loss:.6f}")
            
            # Save best model
            if test_loss < best_loss:
                self._save_best_model(epoch, test_loss)
                best_loss = test_loss
            
            # Update learning rate schedulers
            self._update_schedulers(test_loss)
        
        # Plot training history
        self._plot_training_history(train_losses, test_losses)
        
        # Load best model
        self._load_best_model(train_beta)
    
    def _set_models_mode(self, is_train):
        """
        Set all models to training or evaluation mode
        
        Args:
            is_train: If True, set to training mode, otherwise evaluation mode
        """
        mode_func = lambda model: model.train() if is_train else model.eval()
        
        for model in self.plaq_models:
            mode_func(model)
        for model in self.rect_models:
            mode_func(model)
        if self.use_combined_model:
            for model in self.combine_models:
                mode_func(model)
    
    def _update_schedulers(self, test_loss):
        """
        Update all learning rate schedulers
        
        Args:
            test_loss: Current test loss value
        """
        for scheduler in self.plaq_schedulers:
            scheduler.step(test_loss)
        for scheduler in self.rect_schedulers:
            scheduler.step(test_loss)
        if self.use_combined_model:
            for scheduler in self.combine_schedulers:
                scheduler.step(test_loss)
    
    def _save_best_model(self, epoch, loss):
        """
        Save the best model checkpoint
        
        Args:
            epoch: Current epoch number
            loss: Current loss value
        """
        save_dict = {
            'epoch': epoch,
            'loss': loss,
        }
        # Save state dict for each model
        for i, model in enumerate(self.plaq_models):
            save_dict[f'model_state_dict_plaq_{i}'] = model.state_dict()
        for i, model in enumerate(self.rect_models):
            save_dict[f'model_state_dict_rect_{i}'] = model.state_dict()
        if self.use_combined_model:
            for i, model in enumerate(self.combine_models):
                save_dict[f'model_state_dict_combine_{i}'] = model.state_dict()
        for i, optimizer in enumerate(self.plaq_optimizers):
            save_dict[f'optimizer_state_dict_plaq_{i}'] = optimizer.state_dict()
        for i, optimizer in enumerate(self.rect_optimizers):
            save_dict[f'optimizer_state_dict_rect_{i}'] = optimizer.state_dict()
        if self.use_combined_model:
            for i, optimizer in enumerate(self.combine_optimizers):
                save_dict[f'optimizer_state_dict_combine_{i}'] = optimizer.state_dict()
        torch.save(save_dict, f'models/best_model_L{self.L}_train_beta{self.train_beta}.pt')

    def _plot_training_history(self, train_losses, test_losses):
        """
        Plot and save training history
        
        Args:
            train_losses: List of training losses
            test_losses: List of testing losses
        """
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
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # Load plaq models
        for i, model in enumerate(self.plaq_models):
            model.load_state_dict(checkpoint[f'model_state_dict_plaq_{i}'])
        
        # Load rect models
        for i, model in enumerate(self.rect_models):
            model.load_state_dict(checkpoint[f'model_state_dict_rect_{i}'])
        
        # Load combine models if available
        if self.use_combined_model and f'model_state_dict_combine_0' in checkpoint:
            for i, model in enumerate(self.combine_models):
                model.load_state_dict(checkpoint[f'model_state_dict_combine_{i}'])
        
        print(f"Loaded best models from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}") 