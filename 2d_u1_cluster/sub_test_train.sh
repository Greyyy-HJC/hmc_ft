#!/bin/bash -l

# Use 128 CPU core on 1 node

#PBS -N hmc
#PBS -A L-PARTON
#PBS -l select=1:ncpus=128:mpiprocs=1:ompthreads=128:mem=128gb
#PBS -j oe
#PBS -l walltime=12:00:00
#PBS -o /lcrc/project/L-parton/jinchen/hmc_ft/2d_u1_cluster/log/test_train_joint_jit.log

# switch to the submit directory
WORKDIR=/lcrc/project/L-parton/jinchen/hmc_ft/2d_u1_cluster
cd $WORKDIR

# output node info
echo ' '
echo ">>> PBS_NODEFILE content:"
cat $PBS_NODEFILE
NODES=$(cat $PBS_NODEFILE | uniq | wc -l)
TASKS=$(wc -l < $PBS_NODEFILE)
echo "${NODES}n*${TASKS}t"

# show current time
start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Start time: $start_time"

# Initialize conda properly
source /lcrc/project/L-parton/jinchen/miniconda3/etc/profile.d/conda.sh
conda activate ml

# check python version
python --version

# check python path
echo "Python path: $(which python)"

# run train.py
python train_joint.py --lattice_size 8 --min_beta 3.0 --max_beta 3.0 --n_epochs 4 --batch_size 128 --n_subsets 8 --n_workers 32 --if_check_jac True

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
