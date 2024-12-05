import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftConfig, PeftModel, LoraConfig
from trl import DPOConfig, DPOTrainer
import utils as ut

def main(args):
    # Load dataset
    print("Loading dataset...")
    dataset = ut.load_dataset(args.dataset_path)
    dataset = dataset.train_test_split(test_size=args.test_split)

    # Load PEFT configuration
    print(f"Loading PEFT model configuration from {args.peft_model_id}...")
    config = PeftConfig.from_pretrained(args.peft_model_id)

    # Configure quantization
    bnb_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit)

    # Load base model
    print(f"Loading base model from {config.base_model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",  # Hardcoded
        trust_remote_code=True,  # Hardcoded
    )
    model.config.use_cache = False

    # Load tokenizer
    print(f"Loading tokenizer from {config.base_model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.eos_token = "<|eot_id|>"  # Hardcoded
    tokenizer.pad_token = "<|finetune_right_pad_id|>"  # Hardcoded

    # Load PEFT model
    print(f"Loading PEFT model from {args.peft_model_id}...")
    model = PeftModel.from_pretrained(model, args.peft_model_id, adapter_name="trainable")  # Hardcoded
    model.load_adapter(args.peft_model_id, adapter_name="reference")  # Hardcoded

    tokenizer.chat_template = None

    # Configure training arguments
    training_args = DPOConfig(
        learning_rate=args.learning_rate,
        beta=args.beta,
        loss_type=args.loss_type,
        use_weighting=args.use_weighting,
        rpo_alpha=args.rpo_alpha,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        model_adapter_name="trainable",  # Hardcoded
        ref_adapter_name="reference",  # Hardcoded
        per_device_train_batch_size=args.batch_size,
    )

    # Configure Lora
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'lm_head'],
    )

    # Initialize DPO trainer
    print("Initializing DPO trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
    )

    # Train the model
    print("Starting training...")
    dpo_trainer.train()
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model using PEFT and DPOTrainer.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file (JSONL).")
    parser.add_argument("--test_split", type=float, default=0.15, help="Proportion of dataset to use for testing.")
    parser.add_argument("--peft_model_id", type=str, required=True, help="Path to the PEFT model directory.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Enable 8-bit quantization.")
    parser.add_argument("--output_dir", type=str, default="Llama31_DPO", help="Directory to save the trained model.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Number of steps for logging during training.")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for the AdamW optimizer.")
    parser.add_argument("--beta", type=float, default=0.1, help="Parameter controlling deviation from the reference model.")
    parser.add_argument("--loss_type", type=str, default="sigmoid", help="Type of loss to use for training.")
    parser.add_argument("--use_weighting", action="store_true", help="Enable weighting of the loss.")
    parser.add_argument("--rpo_alpha", type=float, default=None, help="Alpha parameter for the RPO paper.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")

    args = parser.parse_args()
    main(args)
