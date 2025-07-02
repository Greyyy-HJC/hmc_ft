# %%
import os
# Set PyTorch memory management before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import time
import argparse
from hmc_u1_ft import HMC_U1_FT
from field_trans_opt import FieldTransformation
from utils import hmc_summary, set_seed

parser = argparse.ArgumentParser(description='Parameters for Comparison')
parser.add_argument('--lattice_size', type=int, default=16, help='Lattice size (default: 16)')
parser.add_argument('--n_configs', type=int, default=2048, help='Number of configurations (default: 2048)')
parser.add_argument('--beta', type=float, default=6, help='Beta parameter (default: 6)')
parser.add_argument('--train_beta', type=float, default=6, help='Beta parameter for training (default: 6)')
parser.add_argument('--step_size', type=float, default=0.1, help='Step size for HMC (default: 0.1)')
parser.add_argument('--ft_step_size', type=float, default=0.1, help='Step size for FT HMC (default: 0.1)')
parser.add_argument('--max_lag', type=int, default=200, help='Max lag for autocorrelation (default: 200)')
parser.add_argument('--rand_seed', type=int, default=1331, help='Random seed for training (default: 1331)')
parser.add_argument('--save_tag', type=str, default=None, help='Save tag for training (default: None)')
parser.add_argument('--device', type=str, default='cpu', help='Device to use (default: cpu)')

args = parser.parse_args()

# Print all arguments
print("="*60)
print(">>> Arguments:")
print(f"Lattice size: {args.lattice_size}")
print(f"Number of configurations: {args.n_configs}")
print(f"Beta: {args.beta}")
print(f"Training beta: {args.train_beta}")
print(f"Step size: {args.step_size}")
print(f"FT step size: {args.ft_step_size}")
print(f"Max lag: {args.max_lag}")
print(f"Random seed: {args.rand_seed}")
print(f"Save tag: {args.save_tag}")
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

train_beta = args.train_beta
max_lag = args.max_lag

# Initialize timing variables
model_load_time = 0.0
ft_therm_time = 0.0
ft_run_time = 0.0

# Initialize device - use the actual device argument instead of hardcoded value
device = args.device

# Set default type
torch.set_default_dtype(torch.float32)


# %%
#! Field transformation
print(">>> Neural Network Field Transformation HMC Simulation: ")

# initialize the field transformation
n_subsets = 8
n_workers = 0 # * n_workers = 0 is faster
nn_ft = FieldTransformation(lattice_size, device=device, n_subsets=n_subsets, if_check_jac=False, num_workers=n_workers, identity_init=True, save_tag=args.save_tag)

# Load the trained model using the _load_best_model method
model_load_start_time = time.time()
print(">>> Loading trained model")
try:
    nn_ft._load_best_model(train_beta)
    model_load_end_time = time.time()
    model_load_time = model_load_end_time - model_load_start_time
    print(f">>> Model loaded successfully in {model_load_time:.2f} seconds")
except Exception as e:
    print(f">>> Error loading model: {e}")
    raise

field_transformation = nn_ft.field_transformation
compute_jac_logdet = nn_ft.compute_jac_logdet_compiled


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
final_config, plaq_ls, acceptance_rate, topological_charges, hamiltonians = hmc.run(n_iterations, theta_thermalized, store_interval, save_config=False)
ft_run_end_time = time.time()
ft_run_time = ft_run_end_time - ft_run_start_time
print(f">>> Simulation with field transformation completed in {ft_run_time:.2f} seconds")
print(f">>> Total time (Field Transformation HMC): {model_load_time + ft_therm_time + ft_run_time:.2f} seconds")

# Compute autocorrelation of topological charges
hmc_fig = hmc_summary(beta, max_lag, volume, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, therm_acceptance_rate, acceptance_rate)
hmc_fig.savefig(f'plots/comparison_opt_hmc_ft_L{lattice_size}_beta{beta:.1f}_train_beta{train_beta:.1f}_ftstep{ft_step_size:.2f}_{args.save_tag}.pdf', transparent=True)

# Print timing comparison    
print(f">>> Total time (Field Transformation HMC): {model_load_time + ft_therm_time + ft_run_time:.2f} seconds")



# %%
