export HF_TOKEN="<huggingface token>"
export HF_HOME="/nfsd/nldei/girottopie/.cache"
export WANDB_API_KEY="<wandb key>"

accelerate launch --num_processes=1 dpo_finetuning.py \
    --dataset_path ../dataset_generation/data/dpo_dialogues.jsonl \
    --peft_model_id ../llama3.1_finetuning/output/llama3.1_SFT_from_Base/checkpoint-800 \
    --output_dir ./tmp \
    --load_in_8bit \
    --batch_size 2
