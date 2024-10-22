# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class HamiltonSystem:
    def __init__(self, p_dot, u_dot, t_max, delta_t):
        self.p_dot = p_dot
        self.u_dot = u_dot
        self.t_max = t_max
        self.delta_t = delta_t

    def simulate(self, U0, P0):
        t_array = np.arange(0, self.t_max, self.delta_t)
        U_array = np.zeros((len(t_array), *U0.shape))
        P_array = np.zeros((len(t_array), *P0.shape))
        
        U_array[0] = U0
        P_array[0] = P0

        for i in range(1, len(t_array)):
            U_half = U_array[i-1] + 0.5 * self.delta_t * self.u_dot(U_array[i-1], P_array[i-1])
            P_array[i] = P_array[i-1] + self.delta_t * self.p_dot(U_half, P_array[i-1])
            U_array[i] = U_half + 0.5 * self.delta_t * self.u_dot(U_half, P_array[i])

        return t_array, U_array, P_array

    def pick_data(self, t_array, U_array, P_array, interval=1.0):
        indices = np.where(np.mod(t_array, interval) < self.delta_t)[0]
        return t_array[indices], U_array[indices], P_array[indices]

def potential(U):
    return -np.sum(np.cos(U))

def H_U_dot(U, P):
    return P

def H_P_dot(U, P):
    return beta * np.sin(U)

def H_H(U, P):
    return np.sum(P**2 / 2) - beta * np.sum(np.cos(U))

def plot_results(t_sampled, U_sampled, P_sampled, energy_sampled):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(t_sampled, U_sampled.mean(axis=(1,2)))
    axs[0, 0].set_title('Average U vs Time')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Average U')

    axs[0, 1].plot(t_sampled, P_sampled.mean(axis=(1,2)))
    axs[0, 1].set_title('Average P vs Time')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Average P')

    axs[1, 0].plot(t_sampled, energy_sampled)
    axs[1, 0].set_title('Energy vs Time')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Energy')

    axs[1, 1].plot(U_sampled.mean(axis=(1,2)), P_sampled.mean(axis=(1,2)))
    axs[1, 1].set_title('Phase Space')
    axs[1, 1].set_xlabel('Average U')
    axs[1, 1].set_ylabel('Average P')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(U_sampled[-1], cmap='hsv', interpolation='nearest')
    plt.colorbar(label='U')
    plt.title('Final Configuration of U')
    plt.show()

if __name__ == "__main__":
    # Parameters
    beta = 1.0
    t_max = 30.0
    lattice_size = 10
    delta_t = 0.001
    n_samples = 5000

    # Single initial condition
    U0 = np.random.uniform(0, 2*np.pi, size=(lattice_size, lattice_size))
    P0 = np.random.normal(0, 1, size=(lattice_size, lattice_size))

    hamilton_system = HamiltonSystem(H_P_dot, H_U_dot, t_max, delta_t)

    t_array, U_array, P_array = hamilton_system.simulate(U0, P0)
    t_sampled, U_sampled, P_sampled = hamilton_system.pick_data(t_array, U_array, P_array)

    energy_sampled = [H_H(U, P) for U, P in zip(U_sampled, P_sampled)]

    plot_results(t_sampled, U_sampled, P_sampled, energy_sampled)

    print(f">>> Number of force calculations: {len(t_array) - 1}")
    print(f">>> Average final U: {U_sampled[-1].mean():.4f}")

    # Multiple initial conditions
    U0_samples = np.random.uniform(0, 2*np.pi, size=(n_samples, lattice_size, lattice_size))
    P0_samples = np.random.normal(0, 1, size=(n_samples, lattice_size, lattice_size))

    final_U_means = []
    for U0, P0 in tqdm(zip(U0_samples, P0_samples), total=n_samples, desc="Loop in samples"):
        t_array, U_array, P_array = hamilton_system.simulate(U0, P0)
        final_U_means.append(U_array[-1].mean())

    plt.figure(figsize=(10, 6))
    plt.hist(final_U_means, bins=50, color='lightgreen', edgecolor='black', density=True)
    plt.title(f"Distribution of Final U Means (beta = {beta})")
    plt.xlabel("Final U Mean")
    plt.ylabel("Probability Density")
    plt.grid(True, alpha=0.3)
    plt.show()

    avg_final_U_mean = np.mean(final_U_means)
    print(f"Average final U mean: {avg_final_U_mean:.4f}")

# %%
