# %%
import numpy as np
import torch
import math
from tqdm import tqdm
from torch.autograd import grad
import torch.autograd.functional as F
import torch.linalg as linalg

class HMC_U1:
    def __init__(self, lattice_size, beta, n_thermalization_steps, n_steps, step_size, 
                 field_transformation=None, device='cpu', jacobian_interval=10):
        """
        Initialize the HMC_U1 class.
        
        Parameters:
        -----------
        lattice_size : int
            The size of the lattice (assumed to be square).
        beta : float
            The inverse coupling constant.
        n_thermalization_steps : int
            The number of thermalization steps.
        n_steps : int
            The number of leapfrog steps in each HMC trajectory.
        step_size : float
            The step size for each leapfrog step.
        field_transformation : function or None
            A function that applies the field transformation. If None, no transformation is applied.
        device : str
            The device to use for computation ('cpu' or 'cuda').
        jacobian_interval : int
            The interval at which to compute the Jacobian log determinant.
        """
        self.lattice_size = lattice_size
        self.beta = beta
        self.n_thermalization_steps = n_thermalization_steps
        self.n_steps = n_steps
        self.dt = step_size
        self.field_transformation = field_transformation  # This should now be the NNFieldTransformation instance
        self.device = torch.device(device)
        self.jacobian_interval = jacobian_interval
        self.jacobian_cache = None
        self.step_count = 0

        # Set default data type and device
        torch.set_default_dtype(torch.float64)
        torch.set_default_device(self.device)
        torch.manual_seed(1331)

    def initialize(self):
        """
        Initialize the field configuration (theta) uniformly between -pi and pi.
        """
        return torch.empty([2, self.lattice_size, self.lattice_size], 
                           device=self.device).uniform_(-math.pi, math.pi)

    def plaqphase(self, theta):
        """
        Compute the plaquette phase.
        """
        theta0, theta1 = theta[0], theta[1]
        return (theta0 - theta1 - torch.roll(theta0, shifts=-1, dims=1) 
                + torch.roll(theta1, shifts=-1, dims=0))

    def action(self, theta):
        """
        Compute the action without field transformation.
        """
        theta_P = self.plaqphase(theta)  # 计算每个格点的角度
        action_value = (-self.beta) * torch.sum(torch.cos(theta_P))
        
        # 检查 action_value 是否为标量
        assert action_value.dim() == 0, "Action value is not a scalar."
        
        return action_value

    def transformed_action(self, transformed_field):
        """
        Compute the transformed action with the Jacobian term.
        """
        original_field = self.field_transformation(transformed_field)  # 应用场域变换
        original_action = self.action(original_field)  # 计算原始作用量

        jacobian_log_det = self.compute_jacobian_log_det(transformed_field)  # 计算 Jacobian 的对数行列式
        
        transformed_action_value = original_action - jacobian_log_det

        # 检查 transformed_action_value 是否为标量
        assert transformed_action_value.dim() == 0, "Transformed action value is not a scalar."

        return transformed_action_value

    def compute_jacobian_log_det(self, input_tensor):
        """Compute the log determinant of the Jacobian matrix of the transformation."""
        if self.field_transformation is None:
            return 0.0  # If no transformation, log det is 0

        # Use cached Jacobian if available and not expired
        if self.jacobian_cache is not None and self.step_count % self.jacobian_interval != 0:
            return self.jacobian_cache

        # Compute Jacobian using torch.autograd.functional.jacobian
        jacobian = F.jacobian(self.field_transformation, input_tensor)
        
        # Reshape jacobian to 2D matrix
        jacobian_2d = jacobian.view(-1, jacobian.shape[-1])
        
        # Compute singular values
        s = linalg.svdvals(jacobian_2d)
        
        # Compute log determinant as sum of log of singular values
        log_det = torch.sum(torch.log(s))

        # Cache the result
        self.jacobian_cache = log_det

        return log_det

    def force(self, field):
        """
        Compute the force for the current field configuration.
        """
        field = field.detach().requires_grad_(True)
        action_value = (self.transformed_action(field) if self.field_transformation 
                        else self.action(field))

        # 检查 action_value 是否是标量
        if action_value.dim() != 0:
            raise RuntimeError("Action value must be a scalar.")

        force = grad(action_value, field, create_graph=False, retain_graph=False)[0]
        return force

    def leapfrog(self, field, pi):
        """
        Perform leapfrog integration.
        """
        dt = self.dt
        pi = pi - 0.5 * dt * self.force(field)
        for _ in range(self.n_steps):
            field = field + dt * pi
            field = self.regularize(field)
            pi = pi - dt * self.force(field)
        field = field + 0.5 * dt * pi
        return field.detach(), pi

    def metropolis_step(self, field):
        """
        Perform a Metropolis step.
        """
        self.step_count += 1  # Increment step count
        pi = torch.randn_like(field, device=self.device)
        H_old = (self.transformed_action(field) if self.field_transformation 
                 else self.action(field)) + 0.5 * torch.sum(pi ** 2)

        new_field, new_pi = self.leapfrog(field.clone(), pi.clone())
        H_new = (self.transformed_action(new_field) if self.field_transformation 
                 else self.action(new_field)) + 0.5 * torch.sum(new_pi ** 2)

        delta_H = H_new - H_old
        accept_prob = torch.exp(-delta_H)

        if torch.rand([], device=self.device) < accept_prob:
            return new_field, True, H_new.item()
        else:
            return field, False, H_old.item()

    def regularize(self, theta):
        """
        Regularize the angle to be within the range [-pi, pi].
        """
        return theta - 2 * math.pi * torch.floor((theta + math.pi) / (2 * math.pi))

    def thermalize(self):
        """
        Thermalize the system.
        """
        field = self.initialize()
        actions = []
        acceptance_count = 0

        for _ in tqdm(range(self.n_thermalization_steps), desc="Thermalizing"):
            field, accepted, _ = self.metropolis_step(field)
            action = (self.transformed_action(field) if self.field_transformation 
                      else self.action(field)).item()
            actions.append(action)
            if accepted:
                acceptance_count += 1

            # Print progress every 10% of thermalization
            if (_ + 1) % (self.n_thermalization_steps // 10) == 0:
                print(f"Thermalization progress: {(_+1)/self.n_thermalization_steps:.1%}, "
                      f"Current action: {action:.2f}")

        acceptance_rate = acceptance_count / self.n_thermalization_steps
        return field, torch.tensor(actions), acceptance_rate

    def run(self, n_iterations, field):
        """
        Run HMC sampling.
        """
        actions = []
        hamiltonians = []
        acceptance_count = 0
        topological_charges = []

        for _ in tqdm(range(n_iterations), desc="Running HMC"):
            field, accepted, H = self.metropolis_step(field)
            actions.append((self.transformed_action(field) if self.field_transformation 
                            else self.action(field)).item())
            hamiltonians.append(H)
            topological_charges.append(self.topological_charge(field))
            if accepted:
                acceptance_count += 1

        acceptance_rate = acceptance_count / n_iterations
        return (field, torch.tensor(actions), acceptance_rate, 
                torch.tensor(topological_charges), torch.tensor(hamiltonians))

    def topological_charge(self, field):
        """
        Compute the topological charge.
        """
        theta_P = self.plaqphase(field)
        theta_P_wrapped = self.regularize(theta_P)
        Q = torch.floor(0.1 + torch.sum(theta_P_wrapped) / (2 * math.pi))
        return Q.item()
    
# %%
if __name__ == "__main__":
    from nn_model import NNFieldTransformation

    # Simulation parameters
    lattice_size = 16
    beta = 3
    n_thermalization_steps = 100
    n_steps = 100
    step_size = 0.01
    n_iterations = 200
    volume = lattice_size ** 2

    # Initialize HMC with the example field transformation
    device = "cpu"
    hmc = HMC_U1(lattice_size, beta, n_thermalization_steps, n_steps, step_size, field_transformation=None, device=device)

    print(">>> Starting thermalization")
    field, thermalization_actions, thermalization_acceptance_rate = hmc.thermalize()

    print(">>> Running HMC")
    final_field, actions, acceptance_rate, topological_charges, hamiltonians = hmc.run(n_iterations, field)

    # Convert tensors to numpy arrays
    thermalization_actions = thermalization_actions.numpy()
    actions = actions.numpy()
    hamiltonians = hamiltonians.numpy()
    final_field = final_field.numpy()
    topological_charges = topological_charges.numpy()

    print(f"Thermalization acceptance rate: {thermalization_acceptance_rate:.4f}")
    print(f"Acceptance rate: {acceptance_rate:.4f}")
    
# %%
if __name__ == "__main__":
    from utils import plot_results, compute_autocorrelation, compute_autocorrelation_by_def, plaquette_value, calculate_plaquette_from_field
    import numpy as np
    
    # Compute autocorrelation of topological charges
    max_lag = 20
    autocorrelations = compute_autocorrelation(topological_charges, max_lag, beta, volume)
    autocorrelations_by_def = compute_autocorrelation_by_def(topological_charges, max_lag)

    # Plot results
    plot_results(thermalization_actions, actions, topological_charges, hamiltonians, autocorrelations, title_suffix="(Using Infinite Volume Susceptibility)")
    plot_results(thermalization_actions, actions, topological_charges, hamiltonians, autocorrelations_by_def, title_suffix="(Using Autocorrelation by Definition)")

    # Print acceptance rates
    print(f"Thermalization acceptance rate: {thermalization_acceptance_rate:.4f}")
    print(f"Acceptance rate: {acceptance_rate:.4f}")

    # Calculate and print relative change in Hamiltonian (delta H)
    delta_H = np.max(hamiltonians) - np.min(hamiltonians)
    H_mean = np.mean(hamiltonians)
    relative_delta_H = delta_H / H_mean
    print(f"Relative variation of Hamiltonian (delta H / H_mean): {relative_delta_H:.4f}")

    # Calculate and print change in topological charge (delta Q)
    delta_Q = np.max(topological_charges) - np.min(topological_charges)
    print(f"Variation of topological charge (delta Q): {delta_Q:.4f}")

    print(">>> Simulation completed")
    
    
    
# %%
if __name__ == "__main__":
    from nn_model import NNFieldTransformation

    # Simulation parameters
    lattice_size = 16
    beta = 3
    n_thermalization_steps = 100
    n_steps = 100
    step_size = 0.01
    n_iterations = 200
    volume = lattice_size ** 2

    # Initialize HMC with the example field transformation
    device = "cpu"
    hmc = HMC_U1(lattice_size, beta, n_thermalization_steps, n_steps, step_size, field_transformation=None, device=device)
    
    # Train the neural network force
    nn_transformation = NNFieldTransformation(lattice_size, model_type='CNN', device=device)
    nn_transformation.train(hmc, n_iterations=500)

    # Initialize HMC
    hmc = HMC_U1(lattice_size, beta, n_thermalization_steps, n_steps, step_size, field_transformation=nn_transformation, device=device, jacobian_interval=10)

    print(">>> Starting thermalization")
    field, thermalization_actions, thermalization_acceptance_rate = hmc.thermalize()

    print(">>> Running HMC")
    final_field, actions, acceptance_rate, topological_charges, hamiltonians = hmc.run(n_iterations, field)

    # Convert tensors to numpy arrays
    thermalization_actions = thermalization_actions.numpy()
    actions = actions.numpy()
    hamiltonians = hamiltonians.numpy()
    final_field = final_field.numpy()
    topological_charges = topological_charges.numpy()

    print(f"Thermalization acceptance rate: {thermalization_acceptance_rate:.4f}")
    print(f"Acceptance rate: {acceptance_rate:.4f}")
    
# %%
if __name__ == "__main__":
    from utils import plot_results, compute_autocorrelation, compute_autocorrelation_by_def, plaquette_value, calculate_plaquette_from_field
    import numpy as np
    
    # Compute autocorrelation of topological charges
    max_lag = 20
    autocorrelations = compute_autocorrelation(topological_charges, max_lag, beta, volume)
    autocorrelations_by_def = compute_autocorrelation_by_def(topological_charges, max_lag)

    # Plot results
    plot_results(thermalization_actions, actions, topological_charges, hamiltonians, autocorrelations, title_suffix="(Using Infinite Volume Susceptibility)")
    plot_results(thermalization_actions, actions, topological_charges, hamiltonians, autocorrelations_by_def, title_suffix="(Using Autocorrelation by Definition)")

    # Print acceptance rates
    print(f"Thermalization acceptance rate: {thermalization_acceptance_rate:.4f}")
    print(f"Acceptance rate: {acceptance_rate:.4f}")

    # Calculate and print relative change in Hamiltonian (delta H)
    delta_H = np.max(hamiltonians) - np.min(hamiltonians)
    H_mean = np.mean(hamiltonians)
    relative_delta_H = delta_H / H_mean
    print(f"Relative variation of Hamiltonian (delta H / H_mean): {relative_delta_H:.4f}")

    # Calculate and print change in topological charge (delta Q)
    delta_Q = np.max(topological_charges) - np.min(topological_charges)
    print(f"Variation of topological charge (delta Q): {delta_Q:.4f}")

    print(">>> Simulation completed")

# %%
