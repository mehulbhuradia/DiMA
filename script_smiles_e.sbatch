#!/bin/sh
#SBATCH --partition=general
#SBATCH --job-name=sembed
#SBATCH --qos=short         
#SBATCH --time=4:00:00      
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=2   
#SBATCH --mem-per-cpu=100G
#SBATCH --mail-type=END     

module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda activate /tudelft.net/staff-umbrella/Mehul/DiMA/dima

export TMPDIR=./tmp/
export HF_HOME=/tmp/mbhuradia/

srun python sembed.py