import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer
import utils as ut
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
import os
import torch.distributed as dist
import datetime

def main(args):
    # Initialize the process group for distributed training
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout=datetime.timedelta(seconds=7200),
        device_id=torch.device(f"cuda:{os.getenv('LOCAL_RANK')}"), # Assuming correspondence between cuda:<int> and rank <int>
    )
    if args.wandb:
        import wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project="CHATPED_dpo_llama3.1",
            config=args,
        )
        accelerator = Accelerator(
            mixed_precision="no",
            log_with="wandb",
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"
        accelerator = Accelerator(
            mixed_precision="no",
        )

    accelerator.print("cuda version: {}".format(torch.version.cuda))
    accelerator.print("CUDA_HOME: {}".format(os.environ["CUDA_HOME"]))

    # Print arguments
    accelerator.print(args)

    # Load PEFT configuration
    accelerator.print(f"Loading PEFT model configuration from {args.peft_model_id}...")
    config = PeftConfig.from_pretrained(args.peft_model_id)

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load base model
    accelerator.print(f"Loading base model from {config.base_model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,  # Hardcoded
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.enable_input_require_grads() # To avoid error https://github.com/huggingface/trl/issues/731

    # Load tokenizer
    accelerator.print(f"Loading tokenizer from {config.base_model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.eos_token = "<|eot_id|>"  # According to SFT tuning done
    tokenizer.pad_token = "<|finetune_right_pad_id|>"  # According to SFT tuning done

    # Load PEFT model
    accelerator.print(f"Loading PEFT model from {args.peft_model_id}...")
    model = PeftModel.from_pretrained(
        model,
        args.peft_model_id,
        is_trainable=True
    )

    tokenizer.chat_template = None

    # Load dataset
    accelerator.print("Loading Dataset")
    dataset = ut.load_dataset(args.dataset_path)
    # dataset = ut.filter_dataset_mad(dataset, tokenizer)
    # dataset = ut.filter_dataset_by_length(dataset, tokenizer, args.max_len)
    dataset = dataset.shuffle()
    dataset = dataset.train_test_split(test_size=args.test_split)

    # Configure training arguments
    accelerator.print("Configuring training")
    training_args = DPOConfig(
        learning_rate=args.learning_rate,
        beta=args.beta,
        loss_type=args.loss_type,
        use_weighting=args.use_weighting,
        rpo_alpha=args.rpo_alpha,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_acc,
        num_train_epochs=args.epochs,
        label_pad_token_id=tokenizer.pad_token_id,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        precompute_ref_log_probs=True,
        max_length=args.max_len,
        truncation_mode="keep_end",
    )

    # Configure Lora
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "lm_head"]
    )

    # model = get_peft_model(model, peft_config)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]

    # Initialize DPO trainer
    accelerator.print("Initializing DPO trainer...")

    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Train the model
    accelerator.print("Starting training...")
    dpo_trainer.train()
    accelerator.print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model using PEFT and DPOTrainer.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file (JSONL).")
    parser.add_argument("--test_split", type=float, default=0.15, help="Proportion of dataset to use for testing.")
    parser.add_argument("--peft_model_id", type=str, required=True, help="Path to the PEFT model directory.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Enable 8-bit quantization.")
    parser.add_argument("--output_dir", type=str, default="Llama31_DPO", help="Directory to save the trained model.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Number of steps for logging during training.")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for the AdamW optimizer.")
    parser.add_argument("--beta", type=float, default=0.1, help="Parameter controlling deviation from the reference model.")
    parser.add_argument("--loss_type", type=str, default="sigmoid",
                        choices=["sigmoid", "hinge", "ipo", "exo_pair", "nca_pair", "robust",
                                "bco_pair", "sppo_hard", "aot", "aot_pair", "discopop",
                                "apo_zero", "apo_down"],
                        help="Type of loss to use for training. Options: sigmoid (DPO), hinge (SLiC), "
                                "ipo (IPO), exo_pair (EXO), nca_pair (NCA), robust (Robust DPO), "
                                "bco_pair (BCO), sppo_hard (SPPO), aot/aot_pair (AOT), "
                                "discopop (DiscoPOP/LRML), apo_zero/apo_down (APO)")
    parser.add_argument("--use_weighting", action="store_true", help="Enable weighting of the loss.")
    parser.add_argument("--rpo_alpha", type=float, default=None, help="Alpha parameter for the RPO paper.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training per gpu.")
    parser.add_argument("--gradient_acc", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--wandb", action="store_true", help="Enable logging with wandb.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train the model.")
    parser.add_argument("--max_len", type=int, default=1E6, help="Maximum length of the samples (prompt+max(chosen, rejected))")

    args = parser.parse_args()
    main(args)
