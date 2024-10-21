#!/bin/bash

project_dir="/home/gp1108/Code/Thesis/llama3.1_finetuning/training_files/from_instruct"
script_path="${project_dir}/training_script_Instruct.py"
train_data="/home/gp1108/Code/Thesis/llama3.1_finetuning/dataset_preparation/converted_datasets/final_dev_set.json"
dev_data="/home/gp1108/Code/Thesis/llama3.1_finetuning/dataset_preparation/converted_datasets/final_dev_set.json"
output_path="${project_dir}/output"
sif_image_path="/home/gp1108/Code/Thesis/llama3.1_finetuning/training_files/env.sif"
run_name="llama3.1_SFT_from_Instruct"
load_4_bit=False
max_seq_len=120000

HF_TOKEN="<YOUR_HUGGING_FACE_TOKEN>"

cd $project_dir
singularity exec --nv $sif_image_path python3 $script_path \
    --ds_train $train_data \
    --ds_dev $dev_data \
    --run_name $run_name \
    --output_dir $output_path \
    --load_8_bit $load_8_bit \
    --max_seq_len $max_seq_len