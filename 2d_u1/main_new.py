# %%
import numpy as np
import torch
import os
from hmc_u1 import HMC_U1
from hmc_u1_ft import HMC_U1_FT
from nn_model_new import NeuralTransformation
from utils import hmc_summary

def setup_environment(n_threads=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    return device

def run_standard_hmc(params, device):
    print("\n>>> Running Standard HMC Simulation")
    hmc = HMC_U1(params['lattice_size'], params['beta'], 
                 params['n_thermalization_steps'], params['n_steps'], 
                 params['step_size'], device=device)
    
    theta_thermalized, therm_plaq_ls, therm_acceptance_rate = hmc.thermalize()
    final_config, plaq_ls, acceptance_rate, topological_charges, hamiltonians = \
        hmc.run(params['n_iterations'], theta_thermalized)
    
    print(">>> Standard HMC completed")
    return (therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, 
            therm_acceptance_rate, acceptance_rate)

def run_ft_hmc(params, device):
    print("\n>>> Running Neural Network Field Transformation HMC")
    
    # Initialize and train transformation
    nn_transformation = NeuralTransformation(
        lattice_size=params['lattice_size'],
        beta=params['beta'],
        device=device
    )
    loss_history = nn_transformation.train(n_iterations=200, verbose=True)
    
    # Run HMC with transformation
    hmc = HMC_U1_FT(
        params['lattice_size'], params['beta'],
        params['n_thermalization_steps'], params['n_steps'],
        params['step_size'], field_transformation=nn_transformation,
        jacobian_interval=64, device=device
    )
    
    theta_thermalized, therm_plaq_ls, therm_acceptance_rate = hmc.thermalize()
    final_config, plaq_ls, acceptance_rate, topological_charges, hamiltonians = \
        hmc.run(params['n_iterations'], theta_thermalized)
    
    print(">>> FT HMC completed")
    return (therm_plaq_ls, plaq_ls, topological_charges, hamiltonians,
            therm_acceptance_rate, acceptance_rate)

def main():
    # Parameters
    params = {
        'lattice_size': 16,
        'beta': 6,
        'n_thermalization_steps': 30,
        'n_steps': 50,
        'step_size': 0.1,
        'n_iterations': 1024
    }
    
    device = setup_environment()
    max_lag = 20
    volume = params['lattice_size'] ** 2
    
    try:
        # Run standard HMC
        std_results = run_standard_hmc(params, device)
        hmc_summary(params['beta'], max_lag, volume, *std_results)
        
        # Run FT HMC
        ft_results = run_ft_hmc(params, device)
        hmc_summary(params['beta'], max_lag, volume, *ft_results)
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main()

# %%
