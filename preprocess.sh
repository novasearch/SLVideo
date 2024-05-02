#!/bin/bash
#SBATCH --job-name=videolgp
# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1 # Number of cpus/tasks
#SBATCH --mem=2G # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=4:00:00 # No time limit
#SBATCH --gres=gpu:hubgpu:1        #nvidia_a100-pcie-40gb:1  gpu:nvidia_a100-sxm4-40gb:1

# Setup anaconda
eval "$(conda shell.bash hook)"

conda activate base
#cd app

# Run your code
python -m app.preprocess
# flask --app app --debug run -h 0.0.0.0 -p 5432