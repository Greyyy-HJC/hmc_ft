# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.autograd.functional as F
import warnings
import os
import logging

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")

# Set environment variable to control PyTorch log level
os.environ["TORCH_LOGS"] = "ERROR"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

# Configure PyTorch logger, redirect to file
torch_logger = logging.getLogger("torch")
torch_logger.setLevel(logging.ERROR)  # Only show error level logs
# Prevent log propagation to root logger, avoid displaying in console
torch_logger.propagate = False


from utils import plaq_from_field_batch, rect_from_field_batch, get_field_mask, get_plaq_mask, get_rect_mask
from cnn_model_opt import jointCNN

class FieldTransformation:
    """Neural network based field transformation"""
    def __init__(self, lattice_size, device='cpu', n_subsets=8, if_check_jac=False, num_workers=0, identity_init=True, save_tag=None):
        self.L = lattice_size
        self.device = torch.device(device)
        self.n_subsets = n_subsets
        self.if_check_jac = if_check_jac
        self.num_workers = num_workers
        self.train_beta = None # init, will be set in train function
        self.save_tag = save_tag
        
        # Create n_subsets independent models for each subset
        self.models = nn.ModuleList([jointCNN().to(device) for _ in range(n_subsets)])
        
        # * Initialize models to produce nearly identity transformation if identity_init is True
        if identity_init:
            for model in self.models:
                # Initialize all weights to a small non-zero value
                for param in model.parameters():
                    nn.init.normal_(param, mean=0.0, std=0.001)
                    # nn.init.zeros_(param)
        
        self.optimizers = [
            torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            for model in self.models
        ]
        
        self.schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            for optimizer in self.optimizers
        ]
        
        # Use torch.compile to optimize compute-intensive functions
        self._init_compiled_functions()

    def _init_compiled_functions(self):
        """Initialize functions optimized by torch.compile"""
        if hasattr(torch, 'compile'):  # Ensure PyTorch version supports compile
            try:
                # Only compile compute-intensive functions with safer backend option
                # Use 'eager' backend and configure for minimal logs
                compile_options = {
                    "backend": "eager",     # Simple backend, avoid C++ compilation errors
                    "fullgraph": False,     # Do not require full graph compilation
                    "dynamic": True,        # Allow dynamic shapes, reduce recompilation warnings
                }
                
                print("Trying to use torch.compile for optimized computation...")
                self.forward_compiled = torch.compile(self.forward, **compile_options)
                self.ft_phase_compiled = torch.compile(self.ft_phase, **compile_options)
                self.compute_jac_logdet_compiled = torch.compile(self.compute_jac_logdet, **compile_options)
                self.compute_action_compiled = torch.compile(self.compute_action, **compile_options)
                print("Successfully initialized torch.compile")
            except Exception as e:
                print(f"Warning: torch.compile initialization failed: {e}")
                print("Falling back to standard functions")
                self.forward_compiled = self.forward
                self.ft_phase_compiled = self.ft_phase
                self.compute_jac_logdet_compiled = self.compute_jac_logdet
                self.compute_action_compiled = self.compute_action
        else:
            # If PyTorch version does not support compile, use standard functions
            self.forward_compiled = self.forward
            self.ft_phase_compiled = self.ft_phase
            self.compute_jac_logdet_compiled = self.compute_jac_logdet
            self.compute_action_compiled = self.compute_action
            print("torch.compile not available, using standard functions")

    def compute_K0_K1(self, theta, index, plaq, rect):
        """
        OPTIMIZED: Compute K0 and K1 using cached plaq and rect values
        Input: theta with shape [batch_size, 2, L, L], cached plaq and rect
        Output: K0 with shape [batch_size, 4, L, L], K1 with shape [batch_size, 8, L, L]
        """
        batch_size = theta.shape[0]
        
        # Calculate plaq features using cached plaq
        plaq_mask = get_plaq_mask(index, batch_size, self.L).to(self.device)
        plaq_masked = plaq * plaq_mask
        plaq_sin_feature = torch.sin(plaq_masked)
        plaq_cos_feature = torch.cos(plaq_masked)
        plaq_features = torch.stack([plaq_sin_feature, plaq_cos_feature], dim=1)
        
        # Calculate rect features using cached rect
        rect_mask = get_rect_mask(index, batch_size, self.L).to(self.device)
        rect_masked = rect * rect_mask
        rect_sin_feature = torch.sin(rect_masked)
        rect_cos_feature = torch.cos(rect_masked)
        rect_features = torch.cat([rect_sin_feature, rect_cos_feature], dim=1)
        
        # Use joint model to generate K0 and K1
        K0, K1 = self.models[index](plaq_features, rect_features)
        
        return K0, K1

    def ft_phase(self, theta, index):
        """
        Compute the phase factor for field transformation for a specific subset
        OPTIMIZED: Pre-compute all roll operations to avoid redundant computation
        OPTIMIZED: Use sincos fusion to compute sin and cos simultaneously
        """
        batch_size = theta.shape[0]
        
        # OPTIMIZATION: Cache plaq and rect calculations to avoid redundant computation
        plaq = plaq_from_field_batch(theta) # [batch_size, L, L]
        rect = rect_from_field_batch(theta) # [batch_size, 2, L, L]
        
        # ROLL OPTIMIZATION: Pre-compute all rolled versions to avoid redundant roll operations
        # For plaquettes
        plaq_roll_1_2 = torch.roll(plaq, shifts=1, dims=2)    # Used by sin_plaq_dir0_2
        plaq_roll_1_1 = torch.roll(plaq, shifts=1, dims=1)    # Used by sin_plaq_dir1_2
        
        # For rectangles
        rect_dir0 = rect[:, 0, :, :] # [batch_size, L, L]
        rect_dir1 = rect[:, 1, :, :] # [batch_size, L, L]
        
        # Pre-compute all rect roll operations
        rect_dir0_roll_1_1 = torch.roll(rect_dir0, shifts=1, dims=1)
        rect_dir0_roll_1_1_1_2 = torch.roll(rect_dir0, shifts=(1, 1), dims=(1, 2))
        rect_dir0_roll_1_2 = torch.roll(rect_dir0, shifts=1, dims=2)
        
        rect_dir1_roll_1_2 = torch.roll(rect_dir1, shifts=1, dims=2)
        rect_dir1_roll_1_1_1_2 = torch.roll(rect_dir1, shifts=(1, 1), dims=(1, 2))
        rect_dir1_roll_1_1 = torch.roll(rect_dir1, shifts=1, dims=1)
        
        # TRIGONOMETRIC OPTIMIZATION: Compute sin efficiently for plaquettes
        # For the link in direction 0 or 1, there are two related plaquettes, note the group derivative is -sin
        sin_plaq_dir0_1 = -torch.sin(plaq) # [batch_size, L, L]
        sin_plaq_dir0_2 = torch.sin(plaq_roll_1_2)  # Use pre-computed roll
        
        sin_plaq_dir1_1 = torch.sin(plaq)
        sin_plaq_dir1_2 = -torch.sin(plaq_roll_1_1)  # Use pre-computed roll
        
        sin_plaq_stack = torch.stack([sin_plaq_dir0_1, sin_plaq_dir0_2, sin_plaq_dir1_1, sin_plaq_dir1_2], dim=1) # [batch_size, 4, L, L]
        
        # OPTIMIZATION: Pass cached plaq and rect to avoid recomputation
        K0, K1 = self.compute_K0_K1(theta, index, plaq, rect) # [batch_size, 4, L, L], [batch_size, 8, L, L]
        
        # Calculate plaquette phase contribution
        temp = K0 * sin_plaq_stack
        ft_phase_plaq = torch.stack([ 
            temp[:, 0] + temp[:, 1], # dir 0  
            temp[:, 2] + temp[:, 3]  # dir 1
        ], dim=1)  # [batch_size, 2, L, L]
        
        # TRIGONOMETRIC OPTIMIZATION: Pre-compute all sin values for rectangles
        # Stack all angles for vectorized sin computation
        rect_angles = torch.stack([
            rect_dir0_roll_1_1,     # sin_rect_dir0_1 = -sin(this)
            rect_dir0_roll_1_1_1_2, # sin_rect_dir0_2 = sin(this)
            rect_dir0,              # sin_rect_dir0_3 = -sin(this)
            rect_dir0_roll_1_2,     # sin_rect_dir0_4 = sin(this)
            rect_dir1_roll_1_2,     # sin_rect_dir1_1 = sin(this)
            rect_dir1_roll_1_1_1_2, # sin_rect_dir1_2 = -sin(this)
            rect_dir1,              # sin_rect_dir1_3 = sin(this)
            rect_dir1_roll_1_1      # sin_rect_dir1_4 = -sin(this)
        ], dim=1)  # [batch_size, 8, L, L]
        
        # Compute all sin values at once
        sin_rect_values = torch.sin(rect_angles)  # [batch_size, 8, L, L]
        
        # Apply signs according to the original calculation
        sin_rect_signs = torch.tensor([-1, 1, -1, 1, 1, -1, 1, -1], 
                                     device=self.device, dtype=sin_rect_values.dtype)
        sin_rect_stack = sin_rect_values * sin_rect_signs.view(1, 8, 1, 1)
        
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
            theta_curr = theta_curr + self.ft_phase_compiled(theta_curr, index)
            
        return theta_curr
    
    def field_transformation(self, theta):
        """Field transformation function for HMC (single input)"""
        return self.forward(theta.unsqueeze(0)).squeeze(0)

    def field_transformation_compiled(self, theta):
        """Field transformation function for HMC (single input)"""
        return self.forward_compiled(theta.unsqueeze(0)).squeeze(0)

    def inverse(self, theta):
        """
        Transform theta_ori to theta_new sequentially through all subsets
        Uses fixed-point iteration to find the inverse transformation
        """
        theta_curr = theta.clone()
        max_iter = 100
        tol = 1e-6
        
        for index in reversed(range(self.n_subsets)):
            theta_iter = theta_curr.clone()
            
            # Fixed-point iteration to find inverse transformation for this subset
            for i in range(max_iter):
                # Compute the phase for current iteration
                inv_phase = -self.ft_phase_compiled(theta_iter, index)
                
                # Update theta using the inverse phase
                theta_next = theta_curr + inv_phase
                
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
    
    def inverse_field_transformation(self, theta):
        """Inverse field transformation function for HMC (single input)"""
        return self.inverse(theta.unsqueeze(0)).squeeze(0)

    def compute_jac_logdet(self, theta):
        """Compute total log determinant of Jacobian for all subsets
        OPTIMIZED: Pre-compute roll operations to reduce redundant computation
        OPTIMIZED: Vectorized trigonometric function computation
        MEMORY OPTIMIZED: Clear intermediate tensors to prevent OOM"""
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
            
            # ROLL OPTIMIZATION: Pre-compute all rolled versions for Jacobian computation
            plaq_roll_1_2 = torch.roll(plaq, shifts=1, dims=2)
            plaq_roll_1_1 = torch.roll(plaq, shifts=1, dims=1)
            
            rect_dir0_roll_1_1 = torch.roll(rect_dir0, shifts=1, dims=1)
            rect_dir0_roll_1_1_1_2 = torch.roll(rect_dir0, shifts=(1, 1), dims=(1, 2))
            rect_dir0_roll_1_2 = torch.roll(rect_dir0, shifts=1, dims=2)
            
            rect_dir1_roll_1_2 = torch.roll(rect_dir1, shifts=1, dims=2)
            rect_dir1_roll_1_1_1_2 = torch.roll(rect_dir1, shifts=(1, 1), dims=(1, 2))
            rect_dir1_roll_1_1 = torch.roll(rect_dir1, shifts=1, dims=1)
            
            # TRIGONOMETRIC OPTIMIZATION: Vectorized cos computation for plaquettes
            # Stack plaquette angles for vectorized computation
            plaq_angles = torch.stack([
                plaq,           # cos_plaq_dir0_1 = -cos(this)
                plaq_roll_1_2,  # cos_plaq_dir0_2 = -cos(this)
                plaq,           # cos_plaq_dir1_1 = -cos(this)
                plaq_roll_1_1   # cos_plaq_dir1_2 = -cos(this)
            ], dim=1)  # [batch_size, 4, L, L]
            
            # Compute all cos values at once and apply -1 sign
            cos_plaq_stack = -torch.cos(plaq_angles)  # [batch_size, 4, L, L]
            
            #TODO: Clear intermediate variables
            del plaq_angles, plaq_roll_1_2, plaq_roll_1_1
            
            # Get K0, K1 coefficients using cached plaq and rect
            K0, K1 = self.compute_K0_K1(theta_curr, index, plaq, rect) # [batch_size, 4, L, L], [batch_size, 8, L, L]
            
            # Calculate plaquette Jacobian contribution
            temp = K0 * cos_plaq_stack
            plaq_jac_shift = torch.stack([ 
                temp[:, 0] + temp[:, 1],  # dir 0 
                temp[:, 2] + temp[:, 3]   # dir 1
            ], dim=1)  # [batch_size, 2, L, L]
            plaq_jac_shift = plaq_jac_shift * field_mask
            
            #TODO: Clear intermediate variables
            del temp, cos_plaq_stack, K0
            
            # TRIGONOMETRIC OPTIMIZATION: Vectorized cos computation for rectangles
            # Stack all rectangle angles for vectorized computation
            rect_angles = torch.stack([
                rect_dir0_roll_1_1,     # cos_rect_dir0_1 = -cos(this)
                rect_dir0_roll_1_1_1_2, # cos_rect_dir0_2 = -cos(this)
                rect_dir0,              # cos_rect_dir0_3 = -cos(this)
                rect_dir0_roll_1_2,     # cos_rect_dir0_4 = -cos(this)
                rect_dir1_roll_1_2,     # cos_rect_dir1_1 = -cos(this)
                rect_dir1_roll_1_1_1_2, # cos_rect_dir1_2 = -cos(this)
                rect_dir1,              # cos_rect_dir1_3 = -cos(this)
                rect_dir1_roll_1_1      # cos_rect_dir1_4 = -cos(this)
            ], dim=1)  # [batch_size, 8, L, L]
            
            # Compute all cos values at once and apply -1 sign
            cos_rect_stack = -torch.cos(rect_angles)  # [batch_size, 8, L, L]
            
            #TODO: Clear intermediate variables
            del rect_angles, rect_dir0_roll_1_1, rect_dir0_roll_1_1_1_2, rect_dir0_roll_1_2
            del rect_dir1_roll_1_2, rect_dir1_roll_1_1_1_2, rect_dir1_roll_1_1
            
            # Calculate rectangle Jacobian contribution
            temp = K1 * cos_rect_stack
            rect_jac_shift = torch.stack([
                temp[:, 0] + temp[:, 1] + temp[:, 2] + temp[:, 3],  # dir 0
                temp[:, 4] + temp[:, 5] + temp[:, 6] + temp[:, 7]   # dir 1
            ], dim=1)  # [batch_size, 2, L, L]
            rect_jac_shift = rect_jac_shift * field_mask
            
            #TODO: Clear intermediate variables
            del temp, cos_rect_stack, K1
            
            # Accumulate log determinant
            log_det += torch.log(1 + plaq_jac_shift + rect_jac_shift).sum(dim=(1, 2, 3))
            
            #TODO: Clear intermediate variables
            del plaq_jac_shift, rect_jac_shift, field_mask, plaq, rect, rect_dir0, rect_dir1
            
            # Update theta for next subset
            theta_curr = theta_curr + self.ft_phase_compiled(theta_curr, index)
        
        return log_det
    
    def compute_jac_logdet_autograd(self, theta):
        """Compute Jacobian log determinant using autograd (for verification)"""
        theta_single = theta[0].unsqueeze(0)  # Only take the first sample to reduce computation
        jac = F.jacobian(self.forward_compiled, theta_single)
        jac_2d = jac.reshape(theta_single.shape[0], theta_single.numel(), theta_single.numel())
        return torch.logdet(jac_2d)
    
    def compute_action(self, theta, beta):
        """Compute action for given configuration"""
        plaq = plaq_from_field_batch(theta)
        total_action = torch.sum(torch.cos(plaq), dim=(1, 2))
        
        # Apply beta factor
        return -beta * total_action
    
    def compute_force(self, theta, beta, transformed=False):
        """
        OPTIMIZED Compute force (gradient of action) - Vectorized version for better performance
        
        Args:
            theta: Field configuration with shape [batch_size, 2, L, L]
            beta: Coupling constant (float)
            transformed: Whether to compute force in transformed space (bool)
        """
        # Ensure input requires gradients
        if not theta.requires_grad:
            theta = theta.clone().requires_grad_(True)
        
        if transformed:
            # In transformed space, account for the Jacobian
            theta_ori = self.forward_compiled(theta)
            action = self.compute_action_compiled(theta_ori, beta)
            jac_logdet = self.compute_jac_logdet_compiled(theta)
            
            # Verify Jacobian calculation if requested
            if self.if_check_jac:
                jac_logdet_autograd = self.compute_jac_logdet_autograd(theta)
                
                diff = (jac_logdet_autograd[0] - jac_logdet[0]) / jac_logdet[0]
                
                if abs(diff.item()) > 1e-4:
                    print(f"\nWarning: Jacobian log determinant difference = {diff:.2f}")
                    print(">>> Jacobian is not correct!")
                else:
                    print(f"\nJacobian log det (manual): {jac_logdet[0]:.2e}, (autograd): {jac_logdet_autograd[0]:.2e}")
                    print(">>> Jacobian is all good!")
            total_action = action - jac_logdet
        else:
            total_action = self.compute_action_compiled(theta, beta)
        
        # OPTIMIZED: Use vectorized gradient computation instead of the slow per-sample loop!
        # This is 3-5x faster than the original per-sample loop
        total_action = total_action.sum()  # Sum over batch for vectorized computation
        force = torch.autograd.grad(total_action, theta, create_graph=True)[0]
        
        return force  # shape: [batch_size, 2, L, L]
    
    def loss_fn(self, theta_ori):
        """Compute loss function for training"""
        # Transform original configuration to new configuration
        theta_new = self.inverse(theta_ori)
        
        # Compute forces in original and transformed spaces
        force_ori = self.compute_force(theta_new, beta=1) #todo
        force_new = self.compute_force(theta_new, self.train_beta, transformed=True)
        
        # Compute loss using multiple norms
        vol = self.L * self.L
        loss = torch.norm(force_new - force_ori, p=2) / (vol**(1/2)) + \
               torch.norm(force_new - force_ori, p=4) / (vol**(1/4)) + \
               torch.norm(force_new - force_ori, p=6) / (vol**(1/6)) + \
               torch.norm(force_new - force_ori, p=8) / (vol**(1/8))
                
        #TODO: try different loss, variance of force, etc.
        
        return loss

 
    def train_step(self, theta_ori):
        """Perform a single training step for all subsets together"""
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
        for optimizer in self.optimizers:
            optimizer.zero_grad()
    
    def _step_all_optimizers(self):
        """Step all optimizers"""
        for optimizer in self.optimizers:
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
        
        # Print training information
        print(f"\n>>> Training the model at beta = {train_beta}\n")
        
        for epoch in tqdm(range(n_epochs), desc="Training epochs"):
            # Training phase
            self._set_models_mode(True)  # Set models to training mode
            
            epoch_losses = []
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                
            train_loss = np.mean(epoch_losses)
            train_losses.append(train_loss)
            
            # Evaluation phase
            self._set_models_mode(False)  # Set models to evaluation mode
            
            test_losses_epoch = []
            
            for batch in tqdm(test_loader, desc="Evaluating"):
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
        save_dict = {
            'epoch': epoch,
            'loss': loss,
        }
        # Save state dict for each model
        for i, model in enumerate(self.models):
            # Always save the state dict as is, without modifying any prefixes
            # This ensures consistency - we always save with whatever prefix structure the model has
            save_dict[f'model_state_dict_{i}'] = model.state_dict()
                
        for i, optimizer in enumerate(self.optimizers):
            save_dict[f'optimizer_state_dict_{i}'] = optimizer.state_dict()
            
        # make sure the models directory exists
        os.makedirs('models', exist_ok=True)
        if self.save_tag is None:
            torch.save(save_dict, f'models/best_model_opt_L{self.L}_train_beta{self.train_beta}.pt')
        else:
            torch.save(save_dict, f'models/best_model_opt_L{self.L}_train_beta{self.train_beta}_{self.save_tag}.pt')

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
        plt.savefig(f'plots/cnn_opt_loss_L{self.L}_train_beta{self.train_beta}.pdf', transparent=True)
        plt.show()

    def _load_best_model(self, train_beta):
        """
        Load the best model from checkpoint for all subsets
        
        Args:
            train_beta: Beta value used during training
        """
        if self.save_tag is None:
            checkpoint_path = f'models/best_model_opt_L{self.L}_train_beta{train_beta:.1f}.pt'
        else:
            checkpoint_path = f'models/best_model_opt_L{self.L}_train_beta{train_beta:.1f}_{self.save_tag}.pt'
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load models
            for i, model in enumerate(self.models):
                state_dict_key = f'model_state_dict_{i}'
                if state_dict_key in checkpoint:
                    state_dict = checkpoint[state_dict_key]
                    
                    # Check if the current model is wrapped by DataParallel
                    is_data_parallel = isinstance(model, nn.DataParallel)
                    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
                    
                    # If model is not DataParallel but state_dict has 'module.' prefix, remove it
                    if not is_data_parallel and has_module_prefix:
                        print(f"Removing 'module.' prefix from state dict for model {i}")
                        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                        model.load_state_dict(new_state_dict)
                    else:
                        # Direct load when model structure matches the saved state
                        model.load_state_dict(state_dict)
                else:
                    raise KeyError(f"State dict for model {i} not found in checkpoint")
                
            print(f"Loaded best models from epoch {checkpoint['epoch'] + 1} with loss {checkpoint['loss']:.6f}") # +1 because epoch starts from 0
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
