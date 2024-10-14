import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def static_plots(t_sampled, x_sampled, p_sampled, energy_sampled):
        init_energy = energy_sampled[0]
        
        # Create static plots for the Hamiltonian system
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Hamiltonian System Dynamics", fontsize=16)

        plots = [
            (
                axs[0, 0],
                t_sampled,
                x_sampled,
                "Position vs Time",
                "Time t",
                "Position x(t)",
                "blue",
            ),
            (
                axs[0, 1],
                t_sampled,
                p_sampled,
                "Momentum vs Time",
                "Time t",
                "Momentum p(t)",
                "red",
            ),
            (
                axs[1, 0],
                x_sampled,
                p_sampled,
                "Phase Space Plot",
                "Position x",
                "Momentum p",
                "green",
            ),
            (
                axs[1, 1],
                t_sampled,
                energy_sampled,
                "Total Energy vs Time",
                "Time t",
                "Total Energy H(t)",
                "purple",
            ),
        ]
        
        # Create each subplot
        for ax, x, y, title, xlabel, ylabel, color in plots:
            ax.plot(x, y, color=color)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)

        # Adjust energy plot to show conservation
        axs[1, 1].set_ylim(init_energy - 0.1 * init_energy, init_energy + 0.1 * init_energy)
        axs[1, 1].axhline(
            y=init_energy, color="black", linestyle="--", label="Initial Energy"
        )
        axs[1, 1].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def composite_plot(x_sampled, t_sampled, potential_func):
        # Create a composite plot showing particle trajectory and potential energy
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}, sharex=False
        )
        fig.suptitle("Composite Visualization of Hamiltonian System", fontsize=16)

        x_limit_min, x_limit_max = -3 * np.pi, 3 * np.pi
        ax_top.set_xlim(x_limit_min, x_limit_max)
        ax_bottom.set_xlim(x_limit_min, x_limit_max)

        # Plot particle trajectory
        ax_top.plot(x_sampled, t_sampled, color="blue")
        ax_top.set_title("Particle Trajectory")
        ax_top.set_xlabel("Position x(t)")
        ax_top.set_ylabel("Time t")
        ax_top.grid(True)

        # Plot potential energy function
        x_potential = np.linspace(x_limit_min, x_limit_max, 1000)
        V_potential = potential_func(x_potential)
        ax_bottom.plot(x_potential, V_potential, color="black")
        ax_bottom.set_title("Potential Energy V(x) vs Position x")
        ax_bottom.set_xlabel("Position x")
        ax_bottom.set_ylabel("Potential V(x)")
        ax_bottom.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
