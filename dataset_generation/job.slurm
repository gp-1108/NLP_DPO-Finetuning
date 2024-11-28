#!/bin/bash

#SBATCH --job-name dpo_create
#SBATCH --error error_%j.txt
#SBATCH --output output_%j.txt
#SBATCH --mail-user <your_email>
#SBATCH --mail-type ALL
#SBATCH --time 7-24:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 40G

cd /nfsd/nldei/girottopie/NLP_DPO-Finetuning/dataset_generation
echo $PWD

bash ./script.sh