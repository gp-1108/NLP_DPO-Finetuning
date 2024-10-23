#!/bin/bash

#SBATCH --job-name llama_ins
#SBATCH --error error_%j.txt
#SBATCH --output output_%j.txt
#SBATCH --mail-user <your email>
#SBATCH --mail-type ALL
#SBATCH --time 24:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 20G
#SBATCH --gres=gpu:rtx

cd /nfsd/nldei/girottopie/llama3.1_finetuning/training_files/from_instruct
echo $PWD

bash ./script.sh