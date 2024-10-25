# %%
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
    autocorrelation = 1 - Q_diff_squared / (2 * volume * chi_t_inf)
    
    return autocorrelation

def autocorrelation_by_def(Q, delta):
    # round Q to the nearest integer
    Q = np.round(Q).astype(int)
    
    Q_mean = np.mean(Q)
    Q_var = np.var(Q)  # Use np.var for more numerical stability
    
    covariance = np.mean((Q[:-delta] - Q_mean) * (Q[delta:] - Q_mean))
    autocorrelation = covariance / Q_var

    return autocorrelation

Q = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
delta = 1
beta = 1
volume = 10

print(autocorrelation_from_chi(Q, delta, beta, volume))
print(autocorrelation_by_def(Q, delta))
# %%
