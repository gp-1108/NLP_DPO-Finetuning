export HF_TOKEN="<token here>"
export HF_HOME="/nfsd/nldei/girottopie/.cache"
export WANDB_API_KEY="<token here>"

accelerate launch --num_processes=2 --num_machines=1 --mixed_precision=no --dynamo_backend=inductor dpo_finetuning.py \
    --dataset_path ../dataset_generation/data/dpo_dialogues.jsonl \
    --peft_model_id ../llama3.1_finetuning/output/llama3.1_SFT_from_Base/checkpoint-800 \
    --output_dir ./tmp \
    --logging_steps 2 \
    --load_in_8bit \
    --batch_size 1 \
    --gradient_acc 4
