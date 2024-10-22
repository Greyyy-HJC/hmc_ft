import numpy as np
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
        return -self.beta * np.sum(np.cos(U)) + 0.01 * np.sum(U**2)

    def force(self, U):
        return -self.beta * np.sin(U) + 0.02 * U

    def kinetic_energy(self, P):
        return 0.5 * np.sum(P**2)

    def leapfrog(self, U, P):
        P -= 0.5 * self.step_size * self.force(U)
        for _ in range(self.n_steps - 1):
            U += self.step_size * P
            U = np.mod(U, 2 * np.pi)
            P -= self.step_size * self.force(U)
        U += self.step_size * P
        U = np.mod(U, 2 * np.pi)
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
        Q = np.sum(np.sin(dUx) + np.sin(dUy)) / (2 * np.pi)
        return Q

    def run(self, n_iterations, field_transformation=None):
        U = self.initialize()
        actions = []
        hamiltonians = []
        acceptance_rate = 0
        topological_charges = []

        for _ in tqdm(range(n_iterations)):
            if field_transformation:
                U = field_transformation(U)

            U, accepted, H = self.metropolis_step(U)
            actions.append(self.action(U))
            hamiltonians.append(H)
            topological_charges.append(self.topological_charge(U))
            acceptance_rate += accepted

        acceptance_rate /= n_iterations
        return U, np.array(actions), acceptance_rate, np.array(topological_charges), np.array(hamiltonians)