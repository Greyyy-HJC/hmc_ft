# %%
import os
# Set PyTorch memory management before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
from hmc_u1 import HMC_U1
from hmc_u1_ft import HMC_U1_FT
from field_trans import FieldTransformation
from utils import hmc_summary, set_seed


# random seed
rand_seed = 1331
set_seed(rand_seed)

# Parameters
lattice_size = 16
volume = lattice_size ** 2
beta = 6
n_thermalization_steps = 10
n_steps = 10
step_size = 0.1
store_interval = 1
n_iterations = store_interval * 100

# Set default type
torch.set_default_dtype(torch.float32)
device = 'cpu'

# %%
#! No field transformation

print(">>> No Field Transformation HMC Simulation: ")

# Initialize HMC
hmc = HMC_U1(lattice_size, beta, n_thermalization_steps, n_steps, step_size, device=device, if_tune_step_size=False) 

# Thermalize the system
print(">>> Starting thermalization...")
theta_thermalized, therm_plaq_ls, therm_acceptance_rate = hmc.thermalize()

# Run HMC without field transformation
print(">>> Starting simulation...")
config_ls, plaq_ls, acceptance_rate, topological_charges, hamiltonians = hmc.run(n_iterations, theta_thermalized, store_interval, save_config=False)

max_lag = 20
hmc_fig = hmc_summary(beta, max_lag, volume, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, therm_acceptance_rate, acceptance_rate)
hmc_fig.savefig(f'plots/test_hmc.pdf', transparent=True)

# %%
#! Field transformation
print(">>> Neural Network Field Transformation HMC Simulation: ")

# initialize the field transformation
# n_subsets = 8
# n_workers = 0 # * n_workers = 0 is faster
# nn_ft = FieldTransformation(lattice_size, device=device, n_subsets=n_subsets, if_check_jac=False, num_workers=n_workers, identity_init=True)

# for model in nn_ft.models:
#     for param in model.parameters():
#         nn.init.zeros_(param)

# field_transformation = nn_ft.field_transformation
# compute_jac_logdet = nn_ft.compute_jac_logdet

field_transformation = lambda x: x
compute_jac_logdet = lambda x: torch.zeros(1)

hmc = HMC_U1_FT(lattice_size, beta, n_thermalization_steps, n_steps, step_size, field_transformation=field_transformation, compute_jac_logdet=compute_jac_logdet, device=device, if_tune_step_size=False)

# Thermalize the system
print(">>> Starting thermalization with field transformation...")
theta_thermalized, therm_plaq_ls, therm_acceptance_rate = hmc.thermalize()

# Run HMC with field transformation
print(">>> Starting simulation with field transformation...")
final_config, plaq_ls, acceptance_rate, topological_charges, hamiltonians = hmc.run(n_iterations, theta_thermalized, store_interval, save_config=False)

hmc_fig = hmc_summary(beta, max_lag, volume, therm_plaq_ls, plaq_ls, topological_charges, hamiltonians, therm_acceptance_rate, acceptance_rate)
hmc_fig.savefig(f'plots/test_hmc_ft.pdf', transparent=True)

# %%
