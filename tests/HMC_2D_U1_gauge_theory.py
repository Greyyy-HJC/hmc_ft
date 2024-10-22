# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class HMC_U1:
    def __init__(self, lattice_size, beta, n_steps, step_size):
        self.lattice_size = lattice_size
        self.beta = beta
        self.n_steps = n_steps
        self.step_size = step_size

    def initialize(self):
        return np.random.uniform(0, 2 * np.pi, (self.lattice_size, self.lattice_size))

    def action(self, U):
        return -self.beta * np.sum(np.cos(U))

    def force(self, U):
        return -self.beta * np.sin(U)

    def kinetic_energy(self, P):
        return 0.5 * np.sum(P**2)

    def leapfrog(self, U, P):
        P -= 0.5 * self.step_size * self.force(U)
        for _ in range(self.n_steps):
            U += self.step_size * P
            U = np.mod(U + np.pi, 2 * np.pi) - np.pi  # Keep U in [-pi, pi]
            if _ < self.n_steps - 1:
                P -= self.step_size * self.force(U)
        P -= 0.5 * self.step_size * self.force(U)
        return U, P

    def metropolis_step(self, U_old):
        P = np.random.normal(0, 1, U_old.shape)
        H_old = self.action(U_old) + self.kinetic_energy(P)

        U_new, P_new = self.leapfrog(U_old.copy(), P.copy())
        H_new = self.action(U_new) + self.kinetic_energy(P_new)

        if np.random.random() < np.exp(H_old - H_new):
            return U_new, True, H_new
        else:
            return U_old, False, H_old

    def topological_charge(self, U):
        dUx = np.roll(U, -1, axis=0) - U
        dUy = np.roll(U, -1, axis=1) - U
        dUx = np.mod(dUx + np.pi, 2 * np.pi) - np.pi
        dUy = np.mod(dUy + np.pi, 2 * np.pi) - np.pi
        Q = np.sum(np.arctan2(np.sin(dUx + dUy), np.cos(dUx + dUy))) / (2 * np.pi)
        return Q

    def run(self, n_iterations):
        U = self.initialize()
        actions = []
        hamiltonians = []
        acceptance_rate = 0
        topological_charges = []

        for _ in tqdm(range(n_iterations)):
            U, accepted, H = self.metropolis_step(U)
            actions.append(self.action(U))
            hamiltonians.append(H)
            topological_charges.append(self.topological_charge(U))
            acceptance_rate += accepted

        acceptance_rate /= n_iterations
        return U, np.array(actions), acceptance_rate, np.array(hamiltonians), np.array(topological_charges)

def plot_results(actions, hamiltonians, topological_charges, acceptance_rate):
    plt.figure(figsize=(18, 10))

    plt.subplot(221)
    plt.plot(actions)
    plt.title('Action vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Action')

    plt.subplot(222)
    plt.plot(hamiltonians)
    plt.title(f'Hamiltonian vs. Iteration (Acceptance Rate: {acceptance_rate:.2f})')
    plt.xlabel('Iteration')
    plt.ylabel('Hamiltonian')
    plt.axhline(y=hamiltonians[0], color='r', linestyle='--', label='Initial Hamiltonian')
    plt.legend()

    plt.subplot(223)
    plt.plot(topological_charges)
    plt.title('Topological Charge vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Topological Charge')

    plt.subplot(224)
    plt.hist(hamiltonians, bins=50)
    plt.title('Histogram of Hamiltonians')
    plt.xlabel('Hamiltonian')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters
    lattice_size = 10
    beta = 2.0
    n_steps = 10
    step_size = 0.1
    n_iterations = 10000

    hmc = HMC_U1(lattice_size, beta, n_steps, step_size)
    final_config, actions, acceptance_rate, hamiltonians, topological_charges = hmc.run(n_iterations)

    plot_results(actions, hamiltonians, topological_charges, acceptance_rate)

    print(f"Final action: {actions[-1]:.4f}")
    print(f"Acceptance rate: {acceptance_rate:.4f}")
    print(f"Final topological charge: {topological_charges[-1]:.4f}")
    print(f"Hamiltonian variation: {np.std(hamiltonians):.6f}")
    print(f"Hamiltonian range: [{np.min(hamiltonians):.6f}, {np.max(hamiltonians):.6f}]")
    print(f"First 10 Hamiltonians: {hamiltonians[:10]}")

# %%
