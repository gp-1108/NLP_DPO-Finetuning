export HF_TOKEN="<token>"
export HF_HOME="/nfsd/nldei/girottopie/.cache"
export WANDB_API_KEY="<token>"
export CUDA_HOME="/usr/local/cuda-12.4"
export CUDA_VISIBLE_DEVICES=0

apptainer exec \
  --nv \
  --no-home \
  -B /home/girottopie/.cache \
  -B /nfsd/nldei/girottopie \
  env_cuda.sif \
  python3 inference.py \
    --model_path=$1 \
    --temperature=$2 \
    --max_new_tokens=$3 \
    --verbose
