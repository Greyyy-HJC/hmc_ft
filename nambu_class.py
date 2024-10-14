# %%
import numpy as np

class NambuSystem:
    def __init__(self, x_dot, p_dot, r_dot, t_max, delta_t):
        # Initialize the system with given parameters
        self.x_dot = x_dot  # Function for dx/dt
        self.p_dot = p_dot  # Function for dp/dt
        self.r_dot = r_dot  # Function for dr/dt
        self.t_max = t_max  # Maximum simulation time
        self.delta_t = delta_t  # Time step
        self.num_steps = int(t_max / delta_t) + 1  # Number of simulation steps

    def simulate(self, x0, p0, r0):
        # Initialize arrays to store positions, momenta, and time
        x_array = np.zeros(self.num_steps)  # Position array for x
        p_array = np.zeros(self.num_steps)  # Momentum array
        r_array = np.zeros(self.num_steps)  # Position array for r
        t_array = np.linspace(0, self.t_max, self.num_steps)  # Time array

        # Set initial conditions
        x_array[0], p_array[0], r_array[0] = x0, p0, r0

        # Perform simulation using symplectic Euler method
        for i in range(1, self.num_steps):
            # Update momentum, r, and x based on Nambu dynamics equations
            x_array[i] = x_array[i - 1] + self.delta_t * self.x_dot(x_array[i - 1], p_array[i - 1], r_array[i - 1])
            p_array[i] = p_array[i - 1] + self.delta_t * self.p_dot(x_array[i - 1], p_array[i - 1], r_array[i - 1])
            r_array[i] = r_array[i - 1] + self.delta_t * self.r_dot(x_array[i - 1], p_array[i - 1], r_array[i - 1])

        return t_array, x_array, p_array, r_array

    def sample_data(self, t_array, x_array, p_array, r_array):
        # Sample data at regular intervals for plotting
        sample_interval = int(1 / self.delta_t)
        sample_indices = np.arange(0, self.num_steps, sample_interval)
        return [arr[sample_indices] for arr in (t_array, x_array, p_array, r_array)]

# Example usage
if __name__ == "__main__":
    '''
    H = p^2 / 2m + kx^2 / 2 + r^2 / 2m
    G = r^2 / 2m + kx^2 / 4
    
    x_dot = dH/dp * dG/dr - dH/dr * dG/dp
    p_dot = dH/dr * dG/dx - dH/dx * dG/dr
    r_dot = dH/dx * dG/dp - dH/dp * dG/dx
    '''
    
    # Input settings
    k = 1.0  # Spring constant
    m = 1.0  # Mass
    t_max = 30.0  # Maximum time for simulation
    delta_t = 0.0001  # Time step

    # Define the Nambu dynamics equations
    def x_dot(x, p, r):
        return p * r
    
    def p_dot(x, p, r):
        return -(k / 2) * r * x

    def r_dot(x, p, r):
        return -(k / 2) * p * x

    # Initial conditions
    x0 = 0.0
    p0 = 3.0
    r0 = 1.0

    nambu_system = NambuSystem(x_dot, p_dot, r_dot, t_max, delta_t)
    
    t_array, x_array, p_array, r_array = nambu_system.simulate(x0, p0, r0)
    
    t_sampled, x_sampled, p_sampled, r_sampled = nambu_system.sample_data(t_array, x_array, p_array, r_array)

    from plot_class import Plotter
    energy_sampled = 0.5 * (p_sampled**2) / m + 0.5 * (r_sampled**2) / m + k * (x_sampled**2) / 2
    potential = lambda x: k * (x**2) / 2
    
    plotter = Plotter()
    plotter.static_plots(t_sampled, x_sampled, p_sampled, energy_sampled)
    plotter.composite_plot(x_sampled, t_sampled, potential)

# %%
