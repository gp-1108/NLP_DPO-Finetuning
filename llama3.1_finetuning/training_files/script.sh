#!/bin/bash

base_path="/home/gp1108/Code/Thesis/llama3.1_finetuning"
project_dir="${base_path}/training_files"
script_path="${project_dir}/training_script_Base.py"
train_data="${base_path}/dataset_preparation/converted_datasets/final_train_set.json"
dev_data="${base_path}/dataset_preparation/converted_datasets/final_dev_set.json"
output_path="${project_dir}/output"
sif_image_path="${project_dir}/env.sif"
run_name="llama3.1_SFT_from_Instruct"
load_8_bit=True
max_seq_len=120000

export HF_TOKEN="your token"

cd $project_dir
singularity exec --nv -B $base_path $sif_image_path \
    python3 $script_path \
    --ds_train $train_data \
    --ds_dev $dev_data \
    --run_name $run_name \
    --output_dir $output_path \
    --load_8_bit $load_8_bit \
    --max_seq_len $max_seq_len