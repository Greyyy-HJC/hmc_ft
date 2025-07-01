import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from scipy.integrate import quad
from scipy.special import i0, i1

import random

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_field_mask(index, batch_size, L):
    field_mask = torch.zeros((batch_size, 2, L, L), dtype=torch.bool)
    
    if index == 0:
        field_mask[:, 0, 0::2, 0::2] = True
        
    elif index == 1:
        field_mask[:, 0, 0::2, 1::2] = True
        
    elif index == 2:
        field_mask[:, 0, 1::2, 0::2] = True
        
    elif index == 3:
        field_mask[:, 0, 1::2, 1::2] = True
        
    elif index == 4:
        field_mask[:, 1, 0::2, 0::2] = True
        
    elif index == 5:
        field_mask[:, 1, 0::2, 1::2] = True
        
    elif index == 6:
        field_mask[:, 1, 1::2, 0::2] = True
        
    elif index == 7:
        field_mask[:, 1, 1::2, 1::2] = True

    return field_mask

def get_plaq_mask(index, batch_size, L):
    plaq_mask = torch.zeros((batch_size, L, L), dtype=torch.bool)
    
    if index == 0:
        plaq_mask[:, 1::2, :] = True
        
    elif index == 1:
        plaq_mask[:, 1::2, :] = True
        
    elif index == 2:
        plaq_mask[:, 0::2, :] = True
        
    elif index == 3:
        plaq_mask[:, 0::2, :] = True
        
    elif index == 4:
        plaq_mask[:, :, 1::2] = True
        
    elif index == 5:
        plaq_mask[:, :, 0::2] = True
        
    elif index == 6:
        plaq_mask[:, :, 1::2] = True
        
    elif index == 7:
        plaq_mask[:, :, 0::2] = True
        
    return plaq_mask

def get_rect_mask(index, batch_size, L):
    rect_mask = torch.zeros((batch_size, 2, L, L), dtype=torch.bool)
    
    if index == 0:
        rect_mask[:, 1, 1::2, :] = True
        rect_mask[:, 1, 0::2, 1::2] = True
        
    elif index == 1:
        rect_mask[:, 1, 1::2, :] = True
        rect_mask[:, 1, 0::2, 0::2] = True
        
    elif index == 2:
        rect_mask[:, 1, 0::2, :] = True
        rect_mask[:, 1, 1::2, 1::2] = True
        

    elif index == 3:
        rect_mask[:, 1, 0::2, :] = True
        rect_mask[:, 1, 1::2, 0::2] = True
        
    elif index == 4:
        rect_mask[:, 0, :, 1::2] = True
        rect_mask[:, 0, 1::2, 0::2] = True
        
    elif index == 5:
        rect_mask[:, 0, :, 0::2] = True
        rect_mask[:, 0, 1::2, 1::2] = True
        
    elif index == 6:
        rect_mask[:, 0, :, 1::2] = True
        rect_mask[:, 0, 0::2, 0::2] = True
        
    elif index == 7:
        rect_mask[:, 0, :, 0::2] = True
        rect_mask[:, 0, 0::2, 1::2] = True

    return rect_mask
        

def plaq_from_field_batch(theta):
    """
    Calculate the plaquette value for a batch of field configurations.
    Input: theta with shape [batch_size, 2, L, L]
    Output: plaquettes with shape [batch_size, L, L]
    """
    theta0, theta1 = theta[:, 0], theta[:, 1]  # [batch_size, L, L]
    thetaP = theta0 - theta1 - torch.roll(theta0, shifts=-1, dims=2) + torch.roll(theta1, shifts=-1, dims=1)
    
    return thetaP

def rect_from_field_batch(theta):
    """
    Calculate the rectangle value for a batch of field configurations.
    Input: theta with shape [batch_size, 2, L, L]
    Output: rectangles with shape [batch_size, 2, L, L]
    """
    theta0, theta1 = theta[:, 0], theta[:, 1]  # [batch_size, L, L]
    
    rect0 = theta0 + torch.roll(theta0, shifts=-1, dims=1) + torch.roll(theta1, shifts=-2, dims=1) - torch.roll(theta0, shifts=(-1, -1), dims=(1, 2)) - torch.roll(theta0, shifts=-1, dims=2) - theta1
    
    rect1 = theta0 + torch.roll(theta1, shifts=-1, dims=1) + torch.roll(theta1, shifts=(-1, -1), dims=(1, 2)) - torch.roll(theta0, shifts=-2, dims=2) - torch.roll(theta1, shifts=-1, dims=2) - theta1
    
    return torch.stack([rect0, rect1], dim=1)

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
    thetaP_wrapped = regularize(thetaP)
    plaq_mean = torch.mean(torch.cos(thetaP_wrapped))
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

def topo_tensor_from_field(theta):
    """
    Calculate the topological charge for a given field configuration.
    """
    thetaP = plaq_from_field(theta)
    thetaP_wrapped = regularize(thetaP)
    topo = torch.floor(0.1 + thetaP_wrapped / (2 * math.pi))
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

def auto_from_chi_bootstrap(topo, max_lag, beta, volume, n_bootstrap=1000, random_seed=None):
    """
    Compute the autocorrelation function of a sequence of topological charges
    using the method defined in Eq. (7) with bootstrap error estimation.
    
    Parameters:
    topo : numpy.ndarray
        Time series of topological charges.
    max_lag : int
        Maximum lag (i.e., maximum δ value).
    beta : float
        Lattice coupling constant.
    volume : int
        Lattice volume.
    n_bootstrap : int, optional
        Number of bootstrap samples (default: 1000).
    random_seed : int, optional
        Random seed for reproducibility (default: None).
    
    Returns:
    tuple : (autocorrelations, autocorrelations_std)
        autocorrelations : numpy.ndarray
            Mean autocorrelation values for each δ.
        autocorrelations_std : numpy.ndarray
            Standard deviation (error) of autocorrelation values for each δ.
    """
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # round topo to the nearest integer
    topo = np.round(topo).astype(int)
    n_samples = len(topo)
    
    # compute the infinite volume topological susceptibility χ_t^∞(β)
    chi_t_inf = chi_infinity(beta)
    
    # Store bootstrap results
    bootstrap_results = np.zeros((n_bootstrap, max_lag + 1))
    
    # Perform bootstrap resampling
    for i in range(n_bootstrap):
        # Generate bootstrap sample indices (with replacement)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_topo = topo[bootstrap_indices]
        
        # Compute autocorrelation for this bootstrap sample
        autocorrelations = np.zeros(max_lag + 1)
        
        for delta in range(max_lag + 1):
            if delta == 0:
                autocorrelations[delta] = 1.0  # Normalized to 1 at delta=0
            else:
                # compute the square average of the difference of topological charges
                topo_diff_squared = np.mean((bootstrap_topo[:-delta] - bootstrap_topo[delta:]) ** 2)
                
                # compute the autocorrelation function Γ_t(δ)
                autocorrelations[delta] = 1 - topo_diff_squared / (2 * volume * chi_t_inf)
        
        bootstrap_results[i, :] = autocorrelations
    
    # Compute mean and standard deviation
    autocorrelations_mean = np.mean(bootstrap_results, axis=0)
    autocorrelations_std = np.std(bootstrap_results, axis=0)
    
    return autocorrelations_mean, autocorrelations_std

def hmc_summary(beta, max_lag, volume, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, therm_acceptance_rate, acceptance_rate):
    # Compute autocorrelation of topological charges
    # autocor_by_def = auto_by_def(topological_charges, max_lag)
    autocor_from_chi = auto_from_chi(topological_charges, max_lag, beta, volume)


    # Plot results
    # hmc_fig = plot_results(beta, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, autocor_by_def, title_suffix="(Using Auto by Definition)")

    hmc_fig = plot_results(beta, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, autocor_from_chi, title_suffix="(Using Auto from Chi)")

    # Print acceptance rates
    print(f"Thermalization acceptance rate: {therm_acceptance_rate:.4f}")
    print(f"Acceptance rate: {acceptance_rate:.4f}")
    
    return hmc_fig


def plot_results(beta, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, autocorrelations, title_suffix=""):
    fig = plt.figure(figsize=(18, 12))
    fontsize = 18

    plt.subplot(221)
    plt.plot(np.arange(len(therm_plaq_ls)), therm_plaq_ls, label='Thermalization Plaquette', color='blue')
    plt.plot(np.arange(len(plaq_ls)) + len(therm_plaq_ls), plaq_ls, label='Plaquette', color='orange')
    plt.axhline(y=plaq_mean_theory(beta), color='r', linestyle='--', label='Theoretical Plaquette')
    plt.legend(loc='upper right', fontsize=fontsize-2)
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
    # hamiltonians_filtered = hamiltonians[~np.isnan(hamiltonians) & ~np.isinf(hamiltonians)]
    # plt.ylim(np.mean(hamiltonians_filtered) * 0.9, np.mean(hamiltonians_filtered) * 1.1)
    plt.legend(fontsize=fontsize-2, loc='upper right')

    plt.subplot(223)
    plt.plot(topological_charges, marker='o', markersize=3)
    plt.axhline(y=np.mean(topological_charges), color='r', linestyle='--', marker='o', markersize=3, label='Mean Topological Charge')
    plt.title(f'Topological Charge vs. Iteration {title_suffix}', fontsize=fontsize)
    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel('Topological Charge', fontsize=fontsize)
    plt.tick_params(direction="in", top="on", right="on", labelsize=fontsize-2)
    plt.grid(linestyle=":")
    plt.legend(fontsize=fontsize-2, loc='upper right')

    plt.subplot(224)
    plt.plot(range(len(autocorrelations)), autocorrelations, marker='o')
    plt.title('Autocorrelation', fontsize=fontsize)
    plt.xlabel('MDTU', fontsize=fontsize)
    plt.ylabel('Autocorrelation', fontsize=fontsize)
    plt.tick_params(direction="in", top="on", right="on", labelsize=fontsize-2)
    plt.grid(linestyle=":")

    plt.tight_layout()
    plt.show()

    print(">>> Theoretical plaquette: ", plaq_mean_theory(beta))
    print(">>> Mean plaq: ", np.mean(plaq_ls))
    print(">>> Std of mean plaq: ", np.std(plaq_ls) / np.sqrt(len(plaq_ls)))
    
    return fig

def plot_results_with_errors(beta, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, 
                           autocorrelations, autocorrelations_std=None, title_suffix=""):
    """
    Plot results with optional error bars for autocorrelation.
    
    Parameters:
    beta : float
        Lattice coupling constant.
    therm_plaq_ls : list
        List of plaquette values during thermalization.
    plaq_ls : list
        List of plaquette values during simulation.
    topological_charges : list
        List of topological charges.
    hamiltonians : list
        List of Hamiltonian values.
    autocorrelations : numpy.ndarray
        Autocorrelation values.
    autocorrelations_std : numpy.ndarray, optional
        Standard deviation of autocorrelation values for error bars.
    title_suffix : str, optional
        Additional title suffix.
    
    Returns:
    matplotlib.figure.Figure
        The generated figure.
    """
    fig = plt.figure(figsize=(18, 12))
    fontsize = 18

    plt.subplot(221)
    plt.plot(np.arange(len(therm_plaq_ls)), therm_plaq_ls, label='Thermalization Plaquette', color='blue')
    plt.plot(np.arange(len(plaq_ls)) + len(therm_plaq_ls), plaq_ls, label='Plaquette', color='orange')
    plt.axhline(y=plaq_mean_theory(beta), color='r', linestyle='--', label='Theoretical Plaquette')
    plt.legend(loc='upper right', fontsize=fontsize-2)
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
    plt.legend(fontsize=fontsize-2, loc='upper right')

    plt.subplot(223)
    plt.plot(topological_charges, marker='o', markersize=3)
    plt.axhline(y=np.mean(topological_charges), color='r', linestyle='--', marker='o', markersize=3, label='Mean Topological Charge')
    plt.title(f'Topological Charge vs. Iteration {title_suffix}', fontsize=fontsize)
    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel('Topological Charge', fontsize=fontsize)
    plt.tick_params(direction="in", top="on", right="on", labelsize=fontsize-2)
    plt.grid(linestyle=":")
    plt.legend(fontsize=fontsize-2, loc='upper right')

    plt.subplot(224)
    if autocorrelations_std is not None:
        # Plot with error bars
        plt.errorbar(range(len(autocorrelations)), autocorrelations, 
                    yerr=autocorrelations_std, marker='x', capsize=3, capthick=1)
        plt.title('Autocorrelation (with Bootstrap Errors)', fontsize=fontsize)
    else:
        # Plot without error bars
        plt.plot(range(len(autocorrelations)), autocorrelations, marker='x')
        plt.title('Autocorrelation', fontsize=fontsize)
    
    plt.xlabel('MDTU', fontsize=fontsize)
    plt.ylabel('Autocorrelation', fontsize=fontsize)
    plt.tick_params(direction="in", top="on", right="on", labelsize=fontsize-2)
    plt.grid(linestyle=":")

    plt.tight_layout()
    plt.show()

    print(">>> Theoretical plaquette: ", plaq_mean_theory(beta))
    print(">>> Mean plaq: ", np.mean(plaq_ls))
    print(">>> Std of mean plaq: ", np.std(plaq_ls) / np.sqrt(len(plaq_ls)))
    
    if autocorrelations_std is not None:
        print(">>> Autocorrelation errors (bootstrap):")
        for i, (ac, ac_std) in enumerate(zip(autocorrelations, autocorrelations_std)):
            print(f"    δ={i}: {ac:.4f} ± {ac_std:.4f}")
    
    return fig

def hmc_summary_bootstrap(beta, max_lag, volume, therm_plaq_ls, plaq_ls, topological_charges, 
                         hamiltonians, therm_acceptance_rate, acceptance_rate, 
                         n_bootstrap=1000, random_seed=None):
    """
    Compute HMC summary with bootstrap error estimation for autocorrelation.
    
    Parameters:
    beta : float
        Lattice coupling constant.
    max_lag : int
        Maximum lag for autocorrelation.
    volume : int
        Lattice volume.
    therm_plaq_ls : list
        List of plaquette values during thermalization.
    plaq_ls : list
        List of plaquette values during simulation.
    topological_charges : list
        List of topological charges.
    hamiltonians : list
        List of Hamiltonian values.
    therm_acceptance_rate : float
        Acceptance rate during thermalization.
    acceptance_rate : float
        Acceptance rate during simulation.
    n_bootstrap : int, optional
        Number of bootstrap samples (default: 1000).
    random_seed : int, optional
        Random seed for reproducibility (default: None).
    
    Returns:
    matplotlib.figure.Figure
        The generated figure with error bars.
    """
    # Compute autocorrelation with bootstrap error estimation
    autocor_mean, autocor_std = auto_from_chi_bootstrap(
        topological_charges, max_lag, beta, volume, n_bootstrap, random_seed
    )

    # Plot results with error bars
    hmc_fig = plot_results_with_errors(
        beta, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, 
        autocor_mean, autocor_std, title_suffix="(Bootstrap Error Estimation)"
    )

    # Print acceptance rates
    print(f"Thermalization acceptance rate: {therm_acceptance_rate:.4f}")
    print(f"Acceptance rate: {acceptance_rate:.4f}")
    print(f"Bootstrap samples: {n_bootstrap}")
    
    return hmc_fig

