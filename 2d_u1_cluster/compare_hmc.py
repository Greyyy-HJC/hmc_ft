# %%
import os
# Set PyTorch memory management before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import time
import argparse
from hmc_u1 import HMC_U1
from utils import hmc_summary, set_seed

parser = argparse.ArgumentParser(description='Parameters for Comparison')
parser.add_argument('--lattice_size', type=int, default=16, help='Lattice size (default: 16)')
parser.add_argument('--n_configs', type=int, default=2048, help='Number of configurations (default: 2048)')
parser.add_argument('--beta', type=float, default=6, help='Beta parameter (default: 6)')
parser.add_argument('--step_size', type=float, default=0.1, help='Step size for HMC (default: 0.1)')
parser.add_argument('--max_lag', type=int, default=200, help='Max lag for autocorrelation (default: 200)')
parser.add_argument('--rand_seed', type=int, default=1331, help='Random seed for training (default: 1331)')
parser.add_argument('--device', type=str, default='cpu', help='Device to use (default: cpu)')

args = parser.parse_args()

# Print all arguments
print("="*60)
print(">>> Arguments:")
print(f"Lattice size: {args.lattice_size}")
print(f"Number of configurations: {args.n_configs}")
print(f"Beta: {args.beta}")
print(f"Step size: {args.step_size}")
print(f"Max lag: {args.max_lag}")
print(f"Random seed: {args.rand_seed}")
print(f"Device: {args.device}")
print("="*60)

# random seed
set_seed(args.rand_seed)

# Parameters
lattice_size = args.lattice_size
volume = lattice_size ** 2
beta = args.beta
n_thermalization_steps = 200
n_steps = 50
step_size = args.step_size
store_interval = 1
n_iterations = store_interval * args.n_configs
max_lag = args.max_lag

# Initialize timing variables
therm_time = 0.0
run_time = 0.0

# Initialize device - use the actual device argument instead of hardcoded value
device = args.device

# Set default type
torch.set_default_dtype(torch.float32)


# %%
#! No field transformation

print(">>> No Field Transformation HMC Simulation: ")

# Initialize HMC
hmc = HMC_U1(lattice_size, beta, n_thermalization_steps, n_steps, step_size, device=device, if_tune_step_size=True) # todo

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
config_ls, plaq_ls, acceptance_rate, topological_charges, hamiltonians = hmc.run(n_iterations, theta_thermalized, store_interval, save_config=False)
run_end_time = time.time()
run_time = run_end_time - run_start_time
print(f">>> Simulation completed in {run_time:.2f} seconds")
print(f">>> Total time (Standard HMC): {therm_time + run_time:.2f} seconds")


# Compute autocorrelation of topological charges
hmc_fig = hmc_summary(beta, max_lag, volume, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, therm_acceptance_rate, acceptance_rate)
hmc_fig.savefig(f'plots/comparison_hmc_L{lattice_size}_beta{beta:.1f}.pdf', transparent=True)

# Print timing comparison
print(f">>> Total time (Standard HMC): {therm_time + run_time:.2f} seconds")
    



# %%
