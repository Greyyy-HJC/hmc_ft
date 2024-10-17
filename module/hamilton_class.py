# %% 
import numpy as np

class HamiltonSystem:
    def __init__(self, p_dot, x_dot, t_max, delta_t):
        # Initialize the Hamiltonian system with given parameters
        self.p_dot = p_dot  # Function for dp/dt
        self.x_dot = x_dot  # Function for dx/dt
        self.t_max = t_max  # Maximum simulation time
        self.delta_t = delta_t  # Time step
        self.num_steps = int(t_max / delta_t) + 1  # Number of simulation steps

    def simulate(self, x0, p0):
        # Simulate the Hamiltonian system for given initial conditions
        x_array = np.zeros(self.num_steps)  # Array to store positions
        p_array = np.zeros(self.num_steps)  # Array to store momenta
        t_array = np.linspace(0, self.t_max, self.num_steps)  # Time array

        # Set initial conditions
        x_array[0], p_array[0] = x0, p0

        # Perform simulation using symplectic Euler method
        for i in range(1, self.num_steps):
            p_array[i] = p_array[i - 1] + self.delta_t * self.p_dot(x_array[i - 1], p_array[i - 1])
            x_array[i] = x_array[i - 1] + self.delta_t * self.x_dot(x_array[i - 1], p_array[i])

        return t_array, x_array, p_array

    def pick_data(self, t_array, x_array, p_array):
        # Pick data at regular intervals for plotting
        sample_interval = int(1 / self.delta_t)  # sample data points at each second
        sample_indices = np.arange(0, self.num_steps, sample_interval)
        return [arr[sample_indices] for arr in (t_array, x_array, p_array)]


if __name__ == "__main__":
    # Input settings
    k = 1.0  # Spring constant
    m = 1.0  # Mass
    t_max = 30  # Maximum simulation time
    delta_t = 0.01  # Time step
    
    # Example usage of the HamiltonianSystem class
    def p_dot(x, p):
        return -k * x  # For a simple harmonic oscillator, dp/dt = -x

    def x_dot(x, p):
        return p / m  # For a simple harmonic oscillator, dx/dt = p

    # Create Hamiltonian system
    system = HamiltonSystem(p_dot, x_dot, t_max, delta_t)

    # Fixed initial condition
    x0, p0 = 0.0, 3.0

    # Simulate for the fixed initial condition
    t_array, x_array, p_array = system.simulate(x0, p0)

    # Sample the data
    t_sampled, x_sampled, p_sampled = system.pick_data(t_array, x_array, p_array)

    from plot_class import Plotter
    energy_sampled = 0.5 * (p_sampled**2) / m + k * (x_sampled**2) / 2
    potential = lambda x: k * (x**2) / 2
    
    plotter = Plotter()
    plotter.static_plots(t_sampled, x_sampled, p_sampled, energy_sampled)
    plotter.trajectory_plot(x_sampled, t_sampled, potential)



# %%
