import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from scipy.integrate import quad
from scipy.special import i0, i1

def plaq_from_field(theta):
    """
    Calculate the plaquette value for a given field configuration.
    """
    theta0, theta1 = theta[0], theta[1]
    thetaP = theta0 - theta1 - torch.roll(theta0, shifts=-1, dims=1) + torch.roll(theta1, shifts=-1, dims=0)
    return thetaP

def plaq_mean_theory(beta):
    """
    Compute the expected plaquette value <P> = I_1(beta) / I_0(beta),
    where I_n(beta) are the modified Bessel functions of the first kind.
    
    Parameters:
    beta : float
        Lattice coupling constant.
    
    Returns:
    float
        The expected plaquette value.
    """
    # Calculate modified Bessel functions I_1(beta) and I_0(beta)
    I1_f = i1(beta)
    I0_f = i0(beta)

    # Calculate plaquette value
    P_expected = I1_f / I0_f
    return P_expected

def plaq_mean_from_field(theta):
    """
    Calculate the average plaquette value for a given field configuration.
    """
    thetaP = plaq_from_field(theta)
    plaq_mean = torch.mean(torch.cos(thetaP))
    return plaq_mean

def regularize(theta):
    """
    Regularize the plaquette value to the range [-pi, pi).
    """
    theta_wrapped = (theta - math.pi) / (2 * math.pi)
    return 2 * math.pi * (theta_wrapped - torch.floor(theta_wrapped) - 0.5)

def topo_from_field(theta):
    """
    Calculate the topological charge for a given field configuration.
    """
    thetaP = plaq_from_field(theta)
    thetaP_wrapped = regularize(thetaP)
    topo = torch.floor(0.1 + torch.sum(thetaP_wrapped) / (2 * math.pi))
    return topo

def chi_infinity(beta):
    """
    Compute the infinite volume topological susceptibility χ_t^∞(β).
    """
    # define integrand
    def numerator_integrand(phi):
        return (phi / (2 * math.pi)) ** 2 * math.exp(beta * math.cos(phi))

    def denominator_integrand(phi):
        return math.exp(beta * math.cos(phi))

    # numerical integration
    numerator, _ = quad(numerator_integrand, -math.pi, math.pi)
    denominator, _ = quad(denominator_integrand, -math.pi, math.pi)

    return numerator / denominator

def auto_from_chi(topo, max_lag, beta, volume):
    """
    Compute the autocorrelation function of a sequence of topological charges
    using the method defined in Eq. (7).
    
    Parameters:
    topo : numpy.ndarray
        Time series of topological charges.
    max_lag : int
        Maximum lag (i.e., maximum δ value).
    beta : float
        Lattice coupling constant.
    volume : int
        Lattice volume.
    
    Returns:
    autocorrelations : numpy.ndarray
        Autocorrelation values for each δ.
    """
    # round topo to the nearest integer
    topo = np.round(topo).astype(int)
    
    # compute the infinite volume topological susceptibility χ_t^∞(β)
    chi_t_inf = chi_infinity(beta)
    
    autocorrelations = np.zeros(max_lag + 1)
    
    # Compute autocorrelation for each delta
    for delta in range(max_lag + 1):
        if delta == 0:
            autocorrelations[delta] = 1.0  # Normalized to 1 at delta=0
        else:
            # compute the square average of the difference of topological charges
            topo_diff_squared = np.mean((topo[:-delta] - topo[delta:]) ** 2)
            
            # compute the autocorrelation function Γ_t(δ)
            autocorrelations[delta] = 1 - topo_diff_squared / (2 * volume * chi_t_inf)
    
    return autocorrelations

def auto_by_def(topo, max_lag):
    """
    Compute the autocorrelation function of a sequence of topological charges.

    Parameters:
    topo : numpy.ndarray
        Time series of topological charges.
    max_lag : int
        Maximum lag (i.e., maximum δ value).

    Returns:
    autocorrelations : numpy.ndarray
        Autocorrelation values for each δ.
    """
    # round topo to the nearest integer
    topo = np.round(topo).astype(int)
    
    topo_mean = np.mean(topo)
    topo_var = np.var(topo)  # Use np.var for more numerical stability
    
    if topo_var == 0:
        return np.ones(max_lag + 1)  # If variance is 0, return all 1 autocorrelations

    autocorrelations = np.zeros(max_lag + 1)

    # Compute autocorrelation for each delta
    for delta in range(max_lag + 1):
        if delta == 0:
            autocorrelations[delta] = 1.0  # Normalized to 1 at delta=0
        else:
            # Ensure correct slicing to avoid index errors
            covariance = np.mean((topo[:-delta] - topo_mean) * (topo[delta:] - topo_mean))
            autocorrelations[delta] = covariance / topo_var

    return autocorrelations


def hmc_summary(beta, max_lag, volume, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, therm_acceptance_rate, acceptance_rate):
    # Compute autocorrelation of topological charges
    autocor_by_def = auto_by_def(topological_charges, max_lag)
    autocor_from_chi = auto_from_chi(topological_charges, max_lag, beta, volume)


    # Plot results
    plot_results(beta, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, autocor_by_def, title_suffix="(Using Auto by Definition)")

    plot_results(beta, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, autocor_from_chi, title_suffix="(Using Auto from Chi)")


    # Print acceptance rates
    print(f"Thermalization acceptance rate: {therm_acceptance_rate:.4f}")
    print(f"Acceptance rate: {acceptance_rate:.4f}")
    
    return


def plot_results(beta, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, autocorrelations, title_suffix=""):
    plt.figure(figsize=(18, 12))
    fontsize = 18  # Set the font size for labels and titles

    plt.subplot(221)
    plt.plot(np.arange(len(therm_plaq_ls)), therm_plaq_ls, label='Thermalization Plaquette', color='blue')
    plt.plot(np.arange(len(plaq_ls)) + len(therm_plaq_ls), plaq_ls, label='Plaquette', color='orange')
    plt.axhline(y=plaq_mean_theory(beta), color='r', linestyle='--', label='Theoretical Plaquette')
    plt.legend()
    plt.title(f'Plaquette vs. Iteration {title_suffix}', fontsize=fontsize)
    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel('Plaquette', fontsize=fontsize)
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
    plt.ylim(np.mean(hamiltonians) * 0.9, np.mean(hamiltonians) * 1.1)
    plt.legend(fontsize=fontsize-2)

    plt.subplot(223)
    plt.plot(topological_charges, marker='o', markersize=3)
    plt.axhline(y=np.mean(topological_charges), color='r', linestyle='--', marker='o', markersize=3, label='Mean Topological Charge')
    plt.title(f'Topological Charge vs. Iteration {title_suffix}', fontsize=fontsize)
    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel('Topological Charge', fontsize=fontsize)
    plt.tick_params(direction="in", top="on", right="on", labelsize=fontsize-2)
    plt.grid(linestyle=":")
    plt.legend(fontsize=fontsize-2)

    plt.subplot(224)
    plt.plot(range(len(autocorrelations)), autocorrelations, marker='o')
    plt.title('Autocorrelation', fontsize=fontsize)
    plt.xlabel('MDTU', fontsize=fontsize)
    plt.ylabel('Autocorrelation', fontsize=fontsize)
    plt.tick_params(direction="in", top="on", right="on", labelsize=fontsize-2)
    plt.grid(linestyle=":")

    plt.tight_layout()
    plt.show()
