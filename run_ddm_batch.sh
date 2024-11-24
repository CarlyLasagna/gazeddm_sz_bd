#!/bin/bash
#SBATCH --job-name=ddm_m[add model # here]
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5g
#SBATCH --time=14-00:00:00
#SBATCH --account=clasagna0
#SBATCH --partition=standard
#SBATCH --output=parallel_%j.log

pwd; hostname; date

echo "Running STAN ddm [add model # here]"

module load cmake
module load R/4.2

Rscript /nfs/turbo/lsa-clasagna/gazeddm/run_hddm.R