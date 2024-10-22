import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, AutoPeftModelForCausalLM, get_peft_model, prepare_model_for_kbit_training
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from datasets import Dataset
import json
import os
from argparse import ArgumentParser
from typing import Tuple

def parse_args():
    """
    This function parses the command line arguments.
    """
    parser = ArgumentParser(description="Causal SFT of Llama 3.1 on a pedagogical dataset.")

    parser.add_argument("--ds_train", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--ds_dev", type=str, required=True, help="Path to the validation dataset.")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--load_8_bit", type=bool, default=False, help="Whether to load the model in 4-bit precision.")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length for the model.")

    return parser.parse_args()

def load_dataset(path: str) -> Dataset:
    """
    This function loads the dataset from the given path.
    """
    def format_prompt(sample_conversation):
        """
        Formats a sample conversation into a specific prompt structure.

        Args:
            sample_conversation (list of dict): A list of dictionaries where each dictionary represents a turn in the conversation.
                                                Each dictionary should have two keys: 'role' and 'content'.
                                                'role' indicates the speaker's role and 'content' contains the text of the conversation.

        Returns:
            str: A formatted string representing the entire conversation in the specified prompt structure.

        Raises:
            ValueError: If the sample_conversation is empty.
        """
        if len(sample_conversation) == 0:
            raise ValueError("Empty conversation")
        prompt = "<|begin_of_text|>"
        for turn in sample_conversation:
            prompt += f"<|start_header_id|>{turn['role']}<|end_header_id|>\n{turn['content']}<|eot_id|>"
        return prompt
    
    data = json.load(open(path))
    formatted_data = {"prompt": [format_prompt(sample_conversation) for sample_conversation in data]}
    dataset = Dataset.from_dict(formatted_data).shuffle()

    return dataset

def load_model_and_tokenizer(load_in_8bit: bool) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_id="meta-llama/Meta-Llama-3.1-8B"
    lora_peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'lm_head'],
        task_type="CAUSAL_LM",
    )

    bnb_config = BitsAndBytesConfig(
      load_in_8bit=load_in_8bit, bnb_8bit_quant_type="nf4", bnb_8bit_compute_dtype="float16"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=os.environ.get("HF_TOKEN", None),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=os.environ.get("HF_TOKEN", None),
    )
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.eos_token = "<|eot_id|>"
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_peft_config)
    
    return model, tokenizer

def main():
    args = parse_args()

    # Parsing the command line arguments
    train_dataset_path = args.ds_train
    dev_dataset_path = args.ds_dev
    run_name = args.run_name
    output_dir = args.output_dir
    load_in_8bit = args.load_8_bit
    max_seq_length = args.max_seq_len

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(load_in_8bit)

    # Load the datasets
    train_dataset = load_dataset(train_dataset_path)
    dev_dataset = load_dataset(dev_dataset_path)

    # Setting up the training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{run_name}",
        per_device_train_batch_size=16,
        optim="adamw_torch_fused",
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        skip_memory_metrics=False,
        learning_rate=3e-5,
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        num_train_epochs=4,
        save_strategy="steps",
        save_steps=2000,
    )

    # Collator to train on completions only
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # Setting up the trainer
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['prompt'])):
            text = example["prompt"][i]
            output_texts.append(text)
        return output_texts

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        args=training_args,
        eval_dataset=dev_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        max_seq_length=128000,
        formatting_func=formatting_prompts_func,
    )
    os.environ['WANDB_DISABLED'] = 'true'

    # Training the model
    trainer.train()

if __name__ == "__main__":
    main()