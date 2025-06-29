#!/bin/bash -l

# switch to the submit directory
WORKDIR=/eagle/fthmc/run/hmc_ft/2d_u1_cluster
cd $WORKDIR


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
python compare.py --lattice_size 64 --n_configs 10240 --beta 6 --train_beta 2 --step_size 0.1 --ft_step_size 0.1 --rand_seed 2008

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
