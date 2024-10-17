import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def static_plots(dynamics_type, t_sampled, x_sampled, p_sampled, energy_sampled, conserved_G=None):
        # Set font sizes
        plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14})
        
        # Create static plots for the Hamiltonian system
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{dynamics_type} Dynamics", fontsize=16)

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
        ]
        
        # Create each subplot
        for ax, x, y, title, xlabel, ylabel, color in plots:
            ax.plot(x, y, color=color)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)

        # Update the energy plot (axs[1, 1])
        ax_energy = axs[1, 1]
        
        # Calculate relative change in energy
        delta_H = (energy_sampled - energy_sampled[0]) / energy_sampled[0]
        ax_energy.plot(t_sampled, delta_H, color="purple", label="ΔH/H₀")
        
        ax_energy.set_title("Relative Changes in Energy and Conserved Quantity" if conserved_G is not None else "Relative Change in Energy")
        ax_energy.set_xlabel("Time t")
        ax_energy.set_ylabel("Relative Change")
        ax_energy.grid(True)

        if conserved_G is not None:
            # Calculate relative change in conserved quantity G
            delta_G = (conserved_G - conserved_G[0]) / conserved_G[0]
            ax_energy.plot(t_sampled, delta_G, color="orange", label="ΔG/G₀")

        ax_energy.set_ylim(-0.1, 0.1)

        ax_energy.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def trajectory_plot(x_sampled, t_sampled, potential_func):
        # Set font sizes
        plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14})
        
        # Create a composite plot showing particle trajectory and potential energy
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
        )
        fig.suptitle("Particle Trajectory On Potential", fontsize=16)

        x_limit_min, x_limit_max = -3 * np.pi, 3 * np.pi
        ax_top.set_xlim(x_limit_min, x_limit_max)

        # Plot particle trajectory
        ax_top.plot(x_sampled, t_sampled, color="blue")
        ax_top.set_ylabel("Time t")
        ax_top.grid(True)
        ax_top.tick_params(direction="in", top="on", right="on")

        # Plot potential energy function
        x_potential = np.linspace(x_limit_min, x_limit_max, 1000)
        V_potential = potential_func(x_potential)
        ax_bottom.plot(x_potential, V_potential, color="black")
        ax_bottom.set_xlabel("Position x")
        ax_bottom.set_ylabel("Potential V(x)")
        ax_bottom.grid(True)

        # Remove the gap between subplots
        plt.subplots_adjust(hspace=0)

        # Remove x-axis label and ticks from the top subplot
        # ax_top.xaxis.set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
