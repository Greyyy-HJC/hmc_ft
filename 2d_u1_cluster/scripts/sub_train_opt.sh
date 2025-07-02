#!/bin/bash -l

#PBS -N train
#PBS -A fthmc
#PBS -l select=2
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -j oe
#PBS -l walltime=0:30:00
#PBS -o /eagle/fthmc/run/hmc_ft/2d_u1_cluster/logs/train_opt_test.log

# switch to the submit directory
WORKDIR=/eagle/fthmc/run/hmc_ft/2d_u1_cluster
cd $WORKDIR

# output node info
echo ' '
echo ">>> PBS_NODEFILE content:"
cat $PBS_NODEFILE
NODES=$(cat $PBS_NODEFILE | uniq | wc -l)
TASKS=$(wc -l < $PBS_NODEFILE)
echo "${NODES}n*${TASKS}t"

# Get GPU info
nvidia-smi
nvcc --version

# show current time
start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Start time: $start_time"

# Initialize conda properly
source /eagle/fthmc/env/py_env.sh

# check python version
python --version

# check python path
echo "Python path: $(which python)"

# run train.py
torchrun --standalone --nproc_per_node=2 train_opt.py --lattice_size 64 --min_beta 2.0 --max_beta 2.0 --beta_gap 0.5 --n_epochs 16 --batch_size 64 --n_subsets 8 --n_workers 0 --rand_seed 2008 --if_identity_init

# calculate total time
end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "End time: $end_time"

# total time
start_seconds=$(date --date="$start_time" +%s)
end_seconds=$(date --date="$end_time" +%s)
duration=$((end_seconds - start_seconds))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
