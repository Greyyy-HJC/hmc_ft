import numpy as np
from tqdm import tqdm

class HMC_U1:
    def __init__(self, lattice_size, beta, n_thermalization_steps, n_steps, step_size):
        """
        Initialize the HMC_U1 class.

        Parameters:
        -----------
        lattice_size : int
            The size of the lattice (assumed to be square).
        beta : float
            The inverse coupling constant.
        n_steps : int
            The number of leapfrog steps in each HMC trajectory.
        step_size : float
            The step size for each leapfrog step.
        """
        self.lattice_size = lattice_size
        self.beta = beta
        self.n_thermalization_steps = n_thermalization_steps
        self.n_steps = n_steps
        self.step_size = step_size

    def initialize(self):
        # Initialize link angles θ uniformly between 0 and 2π for each direction
        return np.random.uniform(0, 2 * np.pi, (2, self.lattice_size, self.lattice_size))

    def action(self, theta):
        # Compute plaquette angles: P = U0(x,y) * U1(x+1,y) * Udagger0(x,y+1) * Udagger1(x,y), so it corresponds to the calculation of angle as theta_P = theta0(x,y) + theta1(x+1,y) - theta0(x,y+1) - theta1(x,y)
        theta_P = (
            theta[0]
            + np.roll(theta[1], shift=-1, axis=0)
            - np.roll(theta[0], shift=-1, axis=1)
            - theta[1]
        )
        # Compute the action: Re[1 - exp(i * theta_P)] = 1 - cos(theta_P), drop the constant term to get the action as -beta * sum(cos(theta_P))
        action = -self.beta * np.sum(np.cos(theta_P))
        return action

    def force(self, theta):
        # Compute plaquette angles
        theta_P = (
            theta[0]
            + np.roll(theta[1], shift=-1, axis=0)
            - np.roll(theta[0], shift=-1, axis=1)
            - theta[1]
        )
        sin_theta_P = np.sin(theta_P)

        # Force on θ[0] (x-direction links)
        force_theta0 = (
            sin_theta_P
            - np.roll(sin_theta_P, shift=1, axis=1)
        )

        # Force on θ[1] (y-direction links)
        force_theta1 = (
            sin_theta_P
            - np.roll(sin_theta_P, shift=1, axis=0)
        )

        force = -self.beta * np.array([force_theta0, force_theta1])
        return force

    def kinetic_energy(self, pi):
        return 0.5 * np.sum(pi**2)

    def leapfrog(self, theta, pi):
        """
        Perform leapfrog integration to numerically integrate the Hamiltonian equations for coordinates and momenta.

        The leapfrog integration is used to propose new field configurations by simulating the dynamics of the system under a fictitious Hamiltonian. This method is symplectic and time-reversible, which are important properties for maintaining detailed balance in the HMC algorithm.

        Parameters:
        theta (numpy.ndarray): Current field configuration (coordinates).
        pi (numpy.ndarray): Current conjugate momenta.

        Returns:
        tuple: A pair (theta_new, pi_new) containing the updated field configuration and conjugate momenta.
        """
        pi = pi - 0.5 * self.step_size * self.force(theta)
        for _ in range(self.n_steps):
            theta = theta + self.step_size * pi
            theta = np.mod(theta, 2 * np.pi)  # Keep θ within [0, 2π)
            if _ != self.n_steps - 1:
                pi = pi - self.step_size * self.force(theta)
        pi = pi - 0.5 * self.step_size * self.force(theta)
        return theta, pi

    def metropolis_step(self, theta_old):
        # Generate conjugate momenta π from a Gaussian distribution
        pi = np.random.normal(size=theta_old.shape)

        # Compute the initial Hamiltonian
        H_old = self.action(theta_old) + self.kinetic_energy(pi)

        # Perform leapfrog integration
        theta_new, pi_new = self.leapfrog(theta_old.copy(), pi.copy())

        # Reverse the momenta to ensures time reversibility and detailed balance in the Markov chain
        pi_new = -pi_new

        # Compute the new Hamiltonian
        H_new = self.action(theta_new) + self.kinetic_energy(pi_new)

        # Metropolis acceptance criterion
        delta_H = H_new - H_old
        if np.random.uniform() < np.exp(-delta_H):
            # Accept the new configuration
            accepted = True
            return theta_new, accepted, H_new
        else:
            # Reject and keep the old configuration
            accepted = False
            return theta_old, accepted, H_old

    def topological_charge(self, theta):
        """ 
        In the continuous theory, the topological charge is given by Q = 1/(2π) * \int d^2x F_{01}(x) = 1/(2π) * \int d^2x \partial_0 A_1(x) - \partial_1 A_0(x).

        In the discrete theory, the topological charge is given by Q = 1/(2π) * sum(theta_P), where theta_P is the angle deficit of the plaquette P.
        """
        # Compute plaquette angles
        theta_P = (
            theta[0]
            + np.roll(theta[1], shift=-1, axis=0)
            - np.roll(theta[0], shift=-1, axis=1)
            - theta[1]
        )

        # Map the plaquette angles to the interval (-π, π], ensures that contributions from plaquettes correctly represent the physical flux
        theta_P_wrapped = (theta_P + np.pi) % (2 * np.pi) - np.pi

        # Sum over the plaquette angles and divide by 2π
        Q = np.sum(theta_P_wrapped) / (2 * np.pi)
        return Q
    
    def thermalize(self, field_transformation=None):
        theta = self.initialize()
        
        actions = []
        acceptance_count = 0
        
        for _ in tqdm(range(self.n_thermalization_steps), desc="Thermalizing"):
            if field_transformation:
                theta = field_transformation(theta)
            theta, accepted, _ = self.metropolis_step(theta)
            actions.append(self.action(theta))
            if accepted:
                acceptance_count += 1
        
        acceptance_rate = acceptance_count / self.n_thermalization_steps
        return theta, np.array(actions), acceptance_rate

    def run(self, n_iterations, theta_thermalized):
        theta = theta_thermalized
        actions = []
        hamiltonians = []
        acceptance_count = 0
        topological_charges = []

        for _ in tqdm(range(n_iterations), desc="Running HMC"):
            theta, accepted, H = self.metropolis_step(theta)
            actions.append(self.action(theta))
            hamiltonians.append(H)
            topological_charges.append(self.topological_charge(theta))
            if accepted:
                acceptance_count += 1

        acceptance_rate = acceptance_count / n_iterations
        return theta, np.array(actions), acceptance_rate, np.array(topological_charges), np.array(hamiltonians)