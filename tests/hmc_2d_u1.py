# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn

class HMC_U1:
    def __init__(self, lattice_size, beta, n_steps, step_size, use_nn=False):
        self.lattice_size = lattice_size
        self.beta = beta
        self.n_steps = n_steps
        self.step_size = step_size
        self.use_nn = use_nn
        if use_nn:
            self.nn_model = self.build_nn_model()

    def build_nn_model(self):
        # Define a simple feedforward neural network model
        class SimpleNN(nn.Module):
            def __init__(self, input_size, output_size):
                super(SimpleNN, self).__init__()
                self.layer = nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_size)
                )

            def forward(self, x):
                return self.layer(x)

        input_size = self.lattice_size * self.lattice_size
        output_size = input_size
        return SimpleNN(input_size, output_size)

    def initialize(self):
        return np.random.uniform(0, 2 * np.pi, (self.lattice_size, self.lattice_size))

    def action(self, U):
        # Modified action to have multiple peaks and valleys, with slower growth away from zero
        return -self.beta * np.sum(np.cos(U)) + 0.01 * np.sum(U**2)

    def force(self, U):
        # Derivative of the new action
        return -self.beta * np.sin(U) + 0.02 * U

    def kinetic_energy(self, P):
        return 0.5 * np.sum(P**2)

    def leapfrog(self, U, P):
        P -= 0.5 * self.step_size * self.force(U)
        for _ in range(self.n_steps - 1):
            U += self.step_size * P
            U = np.mod(U, 2 * np.pi)  # Ensure U stays within [0, 2*pi]
            P -= self.step_size * self.force(U)
        U += self.step_size * P
        U = np.mod(U, 2 * np.pi)  # Ensure U stays within [0, 2*pi]
        P -= 0.5 * self.step_size * self.force(U)
        return U, P

    def field_transformation(self, U, alpha=0.1):
        # A simple manually adjustable field transformation
        return U + alpha * np.sin(U) + alpha * np.random.uniform(-1, 1, U.shape)

    def nn_field_transformation(self, U):
        # Use neural network to transform the field
        U_flat = U.flatten()
        U_tensor = torch.tensor(U_flat, dtype=torch.float32)
        U_transformed_tensor = self.nn_model(U_tensor)
        U_transformed = U_transformed_tensor.detach().numpy().reshape(U.shape)
        return U_transformed

    def metropolis_step(self, U_old):
        P = np.random.normal(0, 1, U_old.shape)
        H_old = self.action(U_old) + self.kinetic_energy(P)

        U_new, P_new = self.leapfrog(U_old.copy(), P.copy())
        H_new = self.action(U_new) + self.kinetic_energy(P_new)

        # Metropolis acceptance step based on Hamiltonian difference
        if np.random.random() < np.exp(H_old - H_new):
            return U_new, True, H_new
        else:
            return U_old, False, H_old

    def topological_charge(self, U):
        # Calculate the topological charge Q
        dUx = np.roll(U, -1, axis=0) - U  # Forward difference in x direction
        dUy = np.roll(U, -1, axis=1) - U  # Forward difference in y direction
        Q = np.sum(np.sin(dUx) + np.sin(dUy)) / (2 * np.pi)
        return Q

    def run(self, n_iterations, apply_transformation=False):
        U = self.initialize()
        actions = []
        hamiltonians = []
        acceptance_rate = 0
        topological_charges = []

        for _ in tqdm(range(n_iterations)):
            if apply_transformation:
                if self.use_nn:
                    U = self.nn_field_transformation(U)  # Apply neural network transformation
                else:
                    U = self.field_transformation(U)  # Apply manually adjustable transformation

            U, accepted, H = self.metropolis_step(U)
            actions.append(self.action(U))
            hamiltonians.append(H)
            topological_charges.append(self.topological_charge(U))
            acceptance_rate += accepted

        acceptance_rate /= n_iterations
        return U, np.array(actions), acceptance_rate, np.array(topological_charges), np.array(hamiltonians)

def plot_results(actions, topological_charges, hamiltonians, title_suffix=""):
    plt.figure(figsize=(24, 10))

    plt.subplot(231)
    plt.plot(actions)
    plt.title(f'Action vs. Iteration {title_suffix}')
    plt.xlabel('Iteration')
    plt.ylabel('Action')

    plt.subplot(232)
    plt.plot(hamiltonians)
    plt.title(f'Hamiltonian vs. Iteration {title_suffix}')
    plt.xlabel('Iteration')
    plt.ylabel('Hamiltonian')
    plt.axhline(y=np.mean(hamiltonians), color='r', linestyle='--', label='Mean Hamiltonian')
    plt.legend()

    plt.subplot(233)
    plt.plot(topological_charges)
    plt.title(f'Topological Charge vs. Iteration {title_suffix}')
    plt.xlabel('Iteration')
    plt.ylabel('Topological Charge')

    plt.subplot(234)
    plt.hist(actions, bins=30, alpha=0.7, label='Action Histogram')
    plt.title(f'Action Distribution {title_suffix}')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(235)
    plt.hist(hamiltonians, bins=30, alpha=0.7, label='Hamiltonian Histogram')
    plt.title(f'Hamiltonian Distribution {title_suffix}')
    plt.xlabel('Hamiltonian')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(236)
    plt.hist(topological_charges, bins=30, alpha=0.7, label='Topological Charge Histogram')
    plt.title(f'Topological Charge Distribution {title_suffix}')
    plt.xlabel('Topological Charge')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters
    lattice_size = 10
    beta = 2.0
    n_steps = 20  # Increased to further improve Hamiltonian conservation
    step_size = 0.01  # Further reduced to help with Hamiltonian conservation
    n_iterations = 1000

    hmc = HMC_U1(lattice_size, beta, n_steps, step_size)
    # Run HMC without field transformation
    final_config, actions, acceptance_rate, topological_charges, hamiltonians = hmc.run(n_iterations, apply_transformation=False)
    plot_results(actions, topological_charges, hamiltonians, title_suffix="(No Transformation)")

    print(f"Final action (no transformation): {actions[-1]:.4f}")
    print(f"Acceptance rate (no transformation): {acceptance_rate:.4f}")
    print(f"Final topological charge (no transformation): {topological_charges[-1]:.4f}")

    # Run HMC with field transformation (manual transformation)
    final_config_trans, actions_trans, acceptance_rate_trans, topological_charges_trans, hamiltonians_trans = hmc.run(n_iterations, apply_transformation=True)
    plot_results(actions_trans, topological_charges_trans, hamiltonians_trans, title_suffix="(With Manual Transformation)")

    print(f"Final action (with manual transformation): {actions_trans[-1]:.4f}")
    print(f"Acceptance rate (with manual transformation): {acceptance_rate_trans:.4f}")
    print(f"Final topological charge (with manual transformation): {topological_charges_trans[-1]:.4f}")

    # Run HMC with field transformation (neural network transformation)
    hmc_nn = HMC_U1(lattice_size, beta, n_steps, step_size, use_nn=True)
    final_config_nn, actions_nn, acceptance_rate_nn, topological_charges_nn, hamiltonians_nn = hmc_nn.run(n_iterations, apply_transformation=True)
    plot_results(actions_nn, topological_charges_nn, hamiltonians_nn, title_suffix="(With Neural Network Transformation)")

    print(f"Final action (with neural network transformation): {actions_nn[-1]:.4f}")
    print(f"Acceptance rate (with neural network transformation): {acceptance_rate_nn:.4f}")
    print(f"Final topological charge (with neural network transformation): {topological_charges_nn[-1]:.4f}")

# %%
