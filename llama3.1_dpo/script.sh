export HF_TOKEN="<token>"
export HF_HOME="/nfsd/nldei/girottopie/.cache"
export WANDB_API_KEY="<token>"
export CUDA_HOME="/usr/local/cuda-12.6"

# Base variables
sft_model="/nfsd/nldei/girottopie/NLP_DPO-Finetuning/llama3.1_finetuning/output/llama3.1_SFT_from_Base/checkpoint-800"
dataset_path="/nfsd/nldei/girottopie/NLP_DPO-Finetuning/dataset_generation/data/dpo_dialogues.jsonl"
sif_image_path="/nfsd/nldei/girottopie/NLP_DPO-Finetuning/llama3.1_dpo/env_cuda.sif"
output_dir="/nfsd/nldei/girottopie/NLP_DPO-Finetuning/llama3.1_dpo/output"

apptainer exec \
  --nv \
  -B /home/girottopie/.cache \
  -B /home/girottopie/.triton \
  -B /nfsd/nldei/girottopie \
  env_cuda.sif \
    accelerate launch \
      --num_processes=2 \
      --num_machines=1 \
      --mixed_precision=no \
      --dynamo_backend=inductor \
        dpo_finetuning.py \
          --dataset_path $dataset_path \
          --peft_model_id $sft_model \
          --output_dir ./tmp \
          --logging_steps 1 \
          --load_in_8bit \
          --batch_size 1 \
          --gradient_acc 4 \
          --wandb \
          --epochs 3
