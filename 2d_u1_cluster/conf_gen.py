# %%
import torch
import numpy as np
import argparse
from hmc_u1 import HMC_U1
from utils import hmc_summary

# Parse command line arguments
parser = argparse.ArgumentParser(description='HMC simulation parameters')
parser.add_argument('--lattice_size', type=int, default=8,
                    help='Size of the lattice (default: 8)')
parser.add_argument('--beta', type=float, default=3.0,
                    help='Beta parameter (default: 3.0)')
parser.add_argument('--n_thermalization', type=int, default=200,
                    help='Number of thermalization steps (default: 200)')
parser.add_argument('--store_interval', type=int, default=1,
                    help='Interval for storing configurations (default: 1)')
parser.add_argument('--n_configs', type=int, default=2048,
                    help='Number of configurations (default: 2048)')

args = parser.parse_args()

# Parameters
lattice_size = args.lattice_size
volume = lattice_size ** 2
beta = args.beta
n_thermalization_steps = args.n_thermalization
n_steps = 50
step_size = 0.1
store_interval = args.store_interval
n_iterations = store_interval * args.n_configs

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set default type
torch.set_default_dtype(torch.float32)

print(">>> No Field Transformation HMC Simulation: ")

# Initialize HMC
hmc = HMC_U1(lattice_size, beta, n_thermalization_steps, n_steps, step_size, device=device, if_tune_step_size=True)

# Thermalize the system
theta_thermalized, therm_plaq_ls, therm_acceptance_rate = hmc.thermalize()

# Run HMC without field transformation
config_ls, plaq_ls, acceptance_rate, topological_charges, hamiltonians = hmc.run(n_iterations, theta_thermalized, store_interval)

print(">>> Simulation completed")


# Compute autocorrelation of topological charges
max_lag = 20
hmc_fig = hmc_summary(beta, max_lag, volume, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, therm_acceptance_rate, acceptance_rate)
hmc_fig.savefig(f'plots/conf_gen_hmc_L{lattice_size}_beta{beta:.1f}.pdf', transparent=True)

# Save configurations for training
np.save(f'gauges/theta_ori_L{lattice_size}_beta{beta:.1f}.npy', torch.stack(config_ls).detach().cpu().numpy())
