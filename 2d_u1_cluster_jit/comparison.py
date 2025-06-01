# %%
import torch
import os
import time
import argparse
from hmc_u1 import HMC_U1
from hmc_u1_ft import HMC_U1_FT
from field_trans import FieldTransformation
from utils import hmc_summary

parser = argparse.ArgumentParser(description='Parameters for Comparison')
parser.add_argument('--ft_step_size', type=float, default=0.1, help='Step size for FT HMC (default: 0.05)')

args = parser.parse_args()

# Parameters
lattice_size = 16
volume = lattice_size ** 2
beta = 6
n_thermalization_steps = 200
n_steps = 50
step_size = 0.1
store_interval = 1
n_iterations = store_interval * 1024

train_beta = 6

# Initialize device
device = 'cpu'

# Set default type
torch.set_default_dtype(torch.float32)


# %%
#! No field transformation
print(">>> No Field Transformation HMC Simulation: ")

# Initialize HMC
hmc = HMC_U1(lattice_size, beta, n_thermalization_steps, n_steps, step_size, device=device, if_tune_step_size=False) # todo

# Thermalize the system
print(">>> Starting thermalization...")
therm_start_time = time.time()
theta_thermalized, therm_plaq_ls, therm_acceptance_rate = hmc.thermalize()
therm_end_time = time.time()
therm_time = therm_end_time - therm_start_time
print(f">>> Thermalization completed in {therm_time:.2f} seconds")

# Run HMC without field transformation
print(">>> Starting simulation...")
run_start_time = time.time()
config_ls, plaq_ls, acceptance_rate, topological_charges, hamiltonians = hmc.run(n_iterations, theta_thermalized, store_interval)
run_end_time = time.time()
run_time = run_end_time - run_start_time
print(f">>> Simulation completed in {run_time:.2f} seconds")
print(f">>> Total time (Standard HMC): {therm_time + run_time:.2f} seconds")


# Compute autocorrelation of topological charges
max_lag = 20
hmc_fig = hmc_summary(beta, max_lag, volume, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, therm_acceptance_rate, acceptance_rate)
hmc_fig.savefig(f'plots/comparison_hmc_L{lattice_size}_beta{beta:.1f}.pdf', transparent=True)


# %%
#! Field transformation
print(">>> Neural Network Field Transformation HMC Simulation: ")

# initialize the field transformation
n_subsets = 8
n_workers = 8
nn_ft = FieldTransformation(lattice_size, device=device, n_subsets=n_subsets, if_check_jac=False, num_workers=n_workers, identity_init=True)

# Load the trained model using the _load_best_model method
model_load_start_time = time.time()
print(">>> Loading trained model")
try:
    nn_ft._load_best_model(train_beta) # ! None means using the best model after training
    model_load_end_time = time.time()
    model_load_time = model_load_end_time - model_load_start_time
    print(f">>> Model loaded successfully in {model_load_time:.2f} seconds")
except Exception as e:
    print(f">>> Error loading model: {e}")
    raise

field_transformation = nn_ft.field_transformation
compute_jac_logdet = nn_ft.compute_jac_logdet


ft_step_size = args.ft_step_size


hmc = HMC_U1_FT(lattice_size, beta, n_thermalization_steps, n_steps, ft_step_size, field_transformation=field_transformation, compute_jac_logdet=compute_jac_logdet, device=device, if_tune_step_size=False) # todo

# Thermalize the system
print(">>> Starting thermalization with field transformation...")
ft_therm_start_time = time.time()
theta_thermalized, therm_plaq_ls, therm_acceptance_rate = hmc.thermalize()
ft_therm_end_time = time.time()
ft_therm_time = ft_therm_end_time - ft_therm_start_time
print(f">>> Thermalization with field transformation completed in {ft_therm_time:.2f} seconds")

# Run HMC with field transformation
print(">>> Starting simulation with field transformation...")
ft_run_start_time = time.time()
final_config, plaq_ls, acceptance_rate, topological_charges, hamiltonians = hmc.run(n_iterations, theta_thermalized, store_interval)
ft_run_end_time = time.time()
ft_run_time = ft_run_end_time - ft_run_start_time
print(f">>> Simulation with field transformation completed in {ft_run_time:.2f} seconds")
print(f">>> Total time (Field Transformation HMC): {model_load_time + ft_therm_time + ft_run_time:.2f} seconds")

# Compute autocorrelation of topological charges
max_lag = 20
hmc_fig = hmc_summary(beta, max_lag, volume, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, therm_acceptance_rate, acceptance_rate)
hmc_fig.savefig(f'plots/comparison_hmc_ft_L{lattice_size}_beta{beta:.1f}_train_beta{train_beta:.1f}_ftstep{ft_step_size:.2f}.pdf', transparent=True)

# Print timing comparison
print("\n>>> Timing Comparison:")
print(f"Standard HMC - Thermalization: {therm_time:.2f}s, Simulation: {run_time:.2f}s, Total: {therm_time + run_time:.2f}s")
print(f"FT HMC - Model Loading: {model_load_time:.2f}s, Thermalization: {ft_therm_time:.2f}s, Simulation: {ft_run_time:.2f}s, Total: {model_load_time + ft_therm_time + ft_run_time:.2f}s")
print(f"Speed ratio (Standard HMC / FT HMC simulation time): {run_time / ft_run_time:.2f}x")


# %%
