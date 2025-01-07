import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import utils as ut

# Function to format the conversation prompt
def format_prompt(conversation):
    """
    Formats a conversation into the model's prompt structure.

    Args:
        conversation (list of dict): The chat history.

    Returns:
        str: Formatted prompt.
    """
    if len(conversation) == 0:
        return "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n"

    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant specialized in Pedagogy.<|eot_id|>"
    for turn in conversation:
        prompt += f"<|start_header_id|>{turn['role']}<|end_header_id|>\n{turn['content']}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt


# Load the SFT model
def load_sft_model(model_name, device='cuda'):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_negative_answer(prompt, model, tokenizer, device, max_length):
    tokenized = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = tokenized['input_ids']
    response_ids = model.generate(
        input_ids,
        max_new_tokens=max_length,
        temperature=0.8,
        do_sample=True
    )
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate negative answers using an SFT model.")
    parser.add_argument("--local_dpo_path", type=str, required=True, help="Path to the local DPO dataset.")
    parser.add_argument("--sft_model_name", type=str, required=True, help="Path to the SFT model.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSONL file.")
    args = parser.parse_args()

    # Load datasets and model
    print("Loading datasets...")
    local_dpo = ut.load_local_dpo_dataset(args.local_dpo_path)
    argilla_ds = ut.load_argilla_ds()
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_sft_model(args.sft_model_name, device)

    # Open output file
    with open(args.output_file, "w") as f:
        # Iterate through each sample
        for idx, sample in enumerate(argilla_ds):
            prompt = sample["prompt"]
            positive_answer = sample["chosen"]
            max_length = len(tokenizer(positive_answer)["input_ids"])

            # Generate negative answer
            negative_answer = generate_negative_answer(prompt, model, tokenizer, device, max_length)

            # Save result
            output = {
                "prompt": prompt,
                "positive_answer": positive_answer,
                "negative_answer": negative_answer,
                "dataset": "argilla"
            }
            f.write(json.dumps(output) + "\n")

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} samples...")

        for idx, sample in enumerate(local_dpo):
            prompt = sample["prompt"]
            positive_answer = sample["chosen"]
            max_length = len(tokenizer(positive_answer)["input_ids"])

            # Generate negative answer
            negative_answer = generate_negative_answer(prompt, model, tokenizer, device, max_length)

            # Save result
            output = {
                "prompt": prompt,
                "positive_answer": positive_answer,
                "negative_answer": negative_answer,
                "dataset": "local_dpo"
            }
            f.write(json.dumps(output) + "\n")

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} samples...")

    print("Processing complete. Output saved to", args.output_file)

if __name__ == "__main__":
    main()
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset

# Load the datasets
def load_local_dpo_dataset(dataset_path: str) -> Dataset:
    loader = DPODialogueLoader(dataset_path)
    trl_compatible_dataset = from_loader_to_pref_std_dataset(loader)
    return trl_compatible_dataset

def load_argilla_ds():
    train_argilla = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")
    train_argilla = train_argilla.filter(
        lambda r: r["status"] != "tie" and r["chosen_score"] >= 8
    )
    train_argilla = train_argilla.rename_column("input", "prompt")
    accepted_columns = set(["prompt", "chosen", "rejected"])
    all_columns = set(train_argilla.column_names)
    columns_to_remove = all_columns - accepted_columns
    train_argilla = train_argilla.remove_columns(list(columns_to_remove))
    new_ds_dict = {"prompt": [], "chosen": [], "rejected": []}
    for r in train_argilla:
        prompt, chosen, rejected = format_interaction([r["prompt"]], r["chosen"], r["rejected"])
        new_ds_dict["prompt"].append(prompt)
        new_ds_dict["chosen"].append(chosen)
        new_ds_dict["rejected"].append(rejected)
    return Dataset.from_dict(new_ds_dict)

def format_prompt(conversation):
    if len(conversation) == 0:
        return "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n"
    prompt = "<|begin_of_text|>"
    for turn in conversation:
        prompt += f"<|start_header_id|>{turn['role']}<|end_header_id|>\n{turn['content']}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

# Load the SFT model
def load_sft_model(model_name, device='cuda'):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_negative_answer(prompt, model, tokenizer, device, max_length):
    tokenized = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = tokenized['input_ids']
    response_ids = model.generate(
        input_ids,
        max_new_tokens=max_length,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate negative answers using an SFT model.")
    parser.add_argument("--local_dpo_path", type=str, required=True, help="Path to the local DPO dataset.")
    parser.add_argument("--sft_model_name", type=str, required=True, help="Path to the SFT model.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSONL file.")
    args = parser.parse_args()

    # Load datasets and model
    print("Loading datasets...")
    local_dpo = load_local_dpo_dataset(args.local_dpo_path)
    argilla_ds = load_argilla_ds()
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_sft_model(args.sft_model_name, device)

    # Open output file
    with open(args.output_file, "w") as f:
        # Iterate through each sample
        for idx, sample in enumerate(argilla_ds):
            prompt = sample["prompt"]
            positive_answer = sample["chosen"]
            max_length = len(tokenizer(positive_answer)["input_ids"])

            # Generate negative answer
            negative_answer = generate_negative_answer(prompt, model, tokenizer, device, max_length)

            # Save result
            output = {
                "prompt": prompt,
                "positive_answer": positive_answer,
                "negative_answer": negative_answer
            }
            f.write(json.dumps(output) + "\n")

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} samples...")

    print("Processing complete. Output saved to", args.output_file)

if __name__ == "__main__":
    main()
