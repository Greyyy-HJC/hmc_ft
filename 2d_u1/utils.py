import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

def chi_infinity(beta):
    """
    Compute the infinite volume topological susceptibility χ_t^∞(β).
    """
    # define integrand
    def numerator_integrand(phi):
        return (phi / (2 * np.pi)) ** 2 * np.exp(beta * np.cos(phi))

    def denominator_integrand(phi):
        return np.exp(beta * np.cos(phi))

    # numerical integration
    numerator, _ = quad(numerator_integrand, -np.pi, np.pi)
    denominator, _ = quad(denominator_integrand, -np.pi, np.pi)

    return numerator / denominator

def autocorrelation_from_chi(Q, delta, beta, volume):
    """
    Compute the autocorrelation function Γ_t(δ) as defined in Eq. (7).
    
    Parameters:
    Q : numpy.ndarray
        Time series of topological charges.
    delta : int
        Lag δ for the autocorrelation.
    beta : float
        Lattice coupling constant.
    volume : int
        Lattice volume.
    
    Returns:
    float
        Autocorrelation value Γ_t(δ).
    """
    # compute the square average of the difference of topological charges
    Q_diff_squared = np.mean((Q[:-delta] - Q[delta:]) ** 2)

    # compute the infinite volume topological susceptibility χ_t^∞(β)
    chi_t_inf = chi_infinity(beta)

    # compute the autocorrelation function Γ_t(δ)
    gamma_t_delta = 1 - Q_diff_squared / (2 * volume * chi_t_inf)
    
    return gamma_t_delta

def compute_autocorrelation(Q, max_lag, beta, volume):
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
    # round Q to the nearest integer
    Q = np.round(Q).astype(int)
    Q_var = np.var(Q)  # Use np.var for more numerical stability
    
    if Q_var == 0:
        return np.ones(max_lag + 1)  # If variance is 0, return all 1 autocorrelations

    autocorrelations = np.zeros(max_lag + 1)

    # Compute autocorrelation for each delta
    for delta in range(max_lag + 1):
        if delta == 0:
            autocorrelations[delta] = 1.0  # Normalized to 1 at delta=0
        else:
            autocorrelations[delta] = autocorrelation_from_chi(Q, delta, beta, volume)

    return autocorrelations

# def compute_autocorrelation(Q, max_lag):
#     """
#     Compute the autocorrelation function of a sequence of topological charges.

#     Parameters:
#     Q : numpy.ndarray
#         Time series of topological charges.
#     max_lag : int
#         Maximum lag (i.e., maximum δ value).

#     Returns:
#     autocorrelations : numpy.ndarray
#         Autocorrelation values for each δ.
#     """
#     # round Q to the nearest integer
#     Q = np.round(Q).astype(int)
    
#     Q_mean = np.mean(Q)
#     Q_var = np.var(Q)  # Use np.var for more numerical stability
    
#     if Q_var == 0:
#         return np.ones(max_lag + 1)  # If variance is 0, return all 1 autocorrelations

#     autocorrelations = np.zeros(max_lag + 1)

#     # Compute autocorrelation for each delta
#     for delta in range(max_lag + 1):
#         if delta == 0:
#             autocorrelations[delta] = 1.0  # Normalized to 1 at delta=0
#         else:
#             # Ensure correct slicing to avoid index errors
#             covariance = np.mean((Q[:-delta] - Q_mean) * (Q[delta:] - Q_mean))
#             autocorrelations[delta] = covariance / Q_var

#     return autocorrelations


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
