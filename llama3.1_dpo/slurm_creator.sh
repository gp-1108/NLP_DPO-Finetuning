#!/bin/bash

# Directory where SLURM job files will be saved
JOB_DIR="./slurm_jobs"
mkdir -p $JOB_DIR

# Hyperparameters and their possible values
LEARNING_RATES=(1e-5)
BETAS=(0.1 0.3 0.6)
LOSS_TYPES=("sigmoid")
USE_WEIGHTING=(false)
RPO_ALPHAS=("")
EPOCHS=(10)

# Fixed parameters
DATASET_PATH="/nfsd/nldei/girottopie/NLP_DPO-Finetuning/dataset_generation/data/dpo_dialogues.jsonl"
SFT_MODEL="/nfsd/nldei/girottopie/NLP_DPO-Finetuning/llama3.1_finetuning/output/llama3.1_SFT_from_Base/checkpoint-800"
OUTPUT_DIR_BASE="/nfsd/nldei/girottopie/NLP_DPO-Finetuning/llama3.1_dpo/output"
SIF_IMAGE_PATH="/nfsd/nldei/girottopie/NLP_DPO-Finetuning/llama3.1_dpo/env_cuda.sif"
LOGGING_STEPS=1
BATCH_SIZE=1
GRADIENT_ACC=4

# Iterate over hyperparameter combinations
for LR in "${LEARNING_RATES[@]}"; do
  for BETA in "${BETAS[@]}"; do
    for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
      for WEIGHTING in "${USE_WEIGHTING[@]}"; do
        for ALPHA in "${RPO_ALPHAS[@]}"; do
          for EPOCH in "${EPOCHS[@]}"; do

            # Generate a unique name for the job
            JOB_NAME="dpo_lr${LR}_beta${BETA}_loss${LOSS_TYPE}_weight${WEIGHTING}_alpha${ALPHA}_ep${EPOCH}"
            OUTPUT_DIR="${OUTPUT_DIR_BASE}/${JOB_NAME}"

            # Construct the apptainer exec command with line breaks and backslashes
            CMD="apptainer exec --nv \\
  -B /home/girottopie/.cache \\
  -B /home/girottopie/.triton \\
  -B /nfsd/nldei/girottopie \\
  $SIF_IMAGE_PATH \\
    accelerate launch \\
      --num_processes=2 \\
      --num_machines=1 \\
      --mixed_precision=no \\
      --dynamo_backend=inductor \\
        dpo_finetuning.py \\
          --dataset_path $DATASET_PATH \\
          --peft_model_id $SFT_MODEL \\
          --output_dir $OUTPUT_DIR \\
          --logging_steps $LOGGING_STEPS \\
          --load_in_8bit \\
          --batch_size $BATCH_SIZE \\
          --gradient_acc $GRADIENT_ACC \\
          --wandb \\
          --learning_rate $LR \\
          --beta $BETA \\
          --loss_type $LOSS_TYPE \\
          --epochs $EPOCH"

            # Add optional flags conditionally
            [ "$WEIGHTING" = true ] && CMD+=" \\
          --use_weighting"
            [ ! -z "$ALPHA" ] && CMD+=" \\
          --rpo_alpha $ALPHA"

            # Create the SLURM job file
            JOB_FILE="${JOB_DIR}/${JOB_NAME}.slurm"
            cat <<EOL > $JOB_FILE
#!/bin/bash
            
#SBATCH --job-name $JOB_NAME
#SBATCH --error error_%j.txt
#SBATCH --output output_%j.txt
#SBATCH --mail-user pietro.girotto@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 24:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 30G
#SBATCH --gres=gpu:a40:2
            
cd /nfsd/nldei/girottopie/llama3.1_dpo

export HF_TOKEN="<token>"
export HF_HOME="/nfsd/nldei/girottopie/.cache"
export WANDB_API_KEY="<token>"
export CUDA_HOME="/usr/local/cuda-12.6"

$CMD
EOL

            # Submit the SLURM job
            sbatch $JOB_FILE

          done
        done
      done
    done
  done
done
