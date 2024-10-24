import matplotlib.pyplot as plt
import numpy as np

def compute_autocorrelation(Q, max_lag):
    """
    Compute the autocorrelation function of a sequence of topological charges.

    Parameters:
    Q : numpy.ndarray
        Time series of topological charges.
    max_lag : int
        Maximum lag (i.e., maximum δ value).

    Returns:
    autocorrelations : numpy.ndarray
        Autocorrelation values for each δ.
    """
    Q_mean = np.mean(Q)
    Q_var = np.mean((Q - Q_mean) ** 2)

    autocorrelations = np.zeros(max_lag + 1)
    for delta in range(max_lag + 1):
        # Compute <Qτ Qτ+δ>
        covariance = np.mean((Q[:-delta] - Q_mean) * (Q[delta:] - Q_mean)) if delta > 0 else Q_var
        autocorrelations[delta] = covariance / Q_var

    return autocorrelations

def plot_results(thermalization_actions, actions, topological_charges, hamiltonians, autocorrelations, title_suffix=""):
    plt.figure(figsize=(18, 12))
    fontsize = 18  # Set the font size for labels and titles

    plt.subplot(221)
    plt.plot(np.arange(len(thermalization_actions)), thermalization_actions, label='Thermalization Actions', color='blue')
    plt.plot(np.arange(len(actions)) + len(thermalization_actions), actions, label='Actions', color='orange')
    plt.legend()
    plt.title(f'Action vs. Iteration {title_suffix}', fontsize=fontsize)
    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel('Action', fontsize=fontsize)
    plt.tick_params(direction="in", top="on", right="on", labelsize=fontsize-2)
    plt.grid(linestyle=":")

    plt.subplot(222)
    plt.plot(hamiltonians)
    plt.title(f'Hamiltonian vs. Iteration {title_suffix}', fontsize=fontsize)
    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel('Hamiltonian', fontsize=fontsize)
    plt.tick_params(direction="in", top="on", right="on", labelsize=fontsize-2)
    plt.grid(linestyle=":")
    plt.axhline(y=np.mean(hamiltonians), color='r', linestyle='--', label='Mean Hamiltonian')
    plt.legend(fontsize=fontsize-2)

    plt.subplot(223)
    plt.plot(topological_charges)
    plt.title(f'Topological Charge vs. Iteration {title_suffix}', fontsize=fontsize)
    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel('Topological Charge', fontsize=fontsize)
    plt.tick_params(direction="in", top="on", right="on", labelsize=fontsize-2)
    plt.grid(linestyle=":")

    plt.subplot(224)
    plt.plot(range(len(autocorrelations)), autocorrelations, marker='o')
    plt.title('Autocorrelation', fontsize=fontsize)
    plt.xlabel('MDTU', fontsize=fontsize)
    plt.ylabel('Autocorrelation', fontsize=fontsize)
    plt.tick_params(direction="in", top="on", right="on", labelsize=fontsize-2)
    plt.grid(linestyle=":")

    plt.tight_layout()
    plt.show()
