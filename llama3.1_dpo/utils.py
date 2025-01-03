from core.loaders import DPODialogueLoader
from datasets import Dataset, load_dataset
import numpy as np

def format_interaction(previous_turns: list, chosen: str, rejected: str) -> tuple:
    """
    Formats the prompt, chosen, and rejected strings for a given interaction.
    
    Args:
        previous_turns (list): A list of strings representing the previous turns, 
                               alternating between user and assistant. 
                               The last string must be a user question.
        chosen (str): The chosen assistant's answer.
        rejected (str): The rejected assistant's answer.
    
    Returns:
        tuple: A tuple containing formatted prompt, chosen answer, and rejected answer.
    """
    if len(previous_turns) % 2 == 0:
        raise ValueError("The 'previous_turns' list must have an odd length with the last being a user question.")
    
    prompt = "<begin_of_text>"
    for i, turn in enumerate(previous_turns):
        if i % 2 == 0:  # User's turn
            prompt += f"<|start_header_id|>user<|end_header_id|>\n{turn}<|eot_id|>"
        else:  # Assistant's turn
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{turn}<|eot_id|>"

    # Add the assistant's placeholder for the last user question
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

    formatted_chosen = f"{chosen}<|eot_id|>"
    formatted_rejected = f"{rejected}<|eot_id|>"

    return prompt, formatted_chosen, formatted_rejected

def from_loader_to_pref_std_dataset(loader: DPODialogueLoader):
    dataset_dict = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    for i in range(len(loader)):
        dpo_turns = loader.get_dpo_turns_by_dialogue_id(loader[i].id)
        prev_interactions = []
        for i in range(0, len(dpo_turns)-1):
            prev_interactions.append(dpo_turns[i].student_question)
            prev_interactions.append(dpo_turns[i].positive_answer)
        prev_interactions.append(dpo_turns[-1].student_question)
        chosen = dpo_turns[-1].positive_answer
        rejected = dpo_turns[-1].negative_answer

        prompt, chosen, rejected = format_interaction(prev_interactions, chosen, rejected)

        dataset_dict["prompt"].append(prompt)
        dataset_dict["chosen"].append(chosen)
        dataset_dict["rejected"].append(rejected)

    return Dataset.from_dict(dataset_dict)

def load_local_dpo_dataset(dataset_path: str) -> Dataset:
    loader = DPODialogueLoader(dataset_path)
    trl_compatible_dataset = from_loader_to_pref_std_dataset(loader)
    return trl_compatible_dataset

def filter_dataset_mad(dataset, tokenizer, threshold=3.5):
    chosen_lengths = [len(tokenizer.encode(chosen)) for chosen in dataset["chosen"]]
    rejected_lengths = [len(tokenizer.encode(rejected)) for rejected in dataset["rejected"]]
    prompt_lengths = [len(tokenizer.encode(prompt)) for prompt in dataset["prompt"]]

    max_lengths = [max(chosen_lengths[i], rejected_lengths[i], prompt_lengths[i]) for i in range(len(chosen_lengths))]

    # Computing the MAD of the chosen lenghts
    # https://www.statology.org/modified-z-score/
    mad = np.median(np.abs(np.median(max_lengths) - np.array(max_lengths)))
    mod_z_scores = [0.6745 * (x - np.median(max_lengths)) / mad for x in max_lengths]
    # Ideally we should use eliminate < -3.5 and > 3.5 so it would be np.abs(mod_z_scores) > threshold
    # But we don't really care about the samples that are too short as they do fit in memory
    mask = np.array(mod_z_scores) < threshold

    # Creating a new dataset with the filtered samples
    new_dataset = {
        "prompt": [dataset["prompt"][i] for i in range(len(dataset["prompt"])) if mask[i]],
        "chosen": [dataset["chosen"][i] for i in range(len(dataset["chosen"])) if mask[i]],
        "rejected": [dataset["rejected"][i] for i in range(len(dataset["rejected"])) if mask[i]]
    }

    return Dataset.from_dict(new_dataset)

def filter_dataset_by_length(dataset, tokenizer, max_length=None):
    if not max_length:
        return dataset

    # Vectorize tokenization
    chosen_lengths = np.array([len(tokenizer.encode(chosen)) for chosen in dataset["chosen"]])
    rejected_lengths = np.array([len(tokenizer.encode(rejected)) for rejected in dataset["rejected"]])
    prompt_lengths = np.array([len(tokenizer.encode(prompt)) for prompt in dataset["prompt"]])

    # Vectorized max operation
    max_lengths = np.maximum(chosen_lengths, rejected_lengths) + prompt_lengths

    # Create mask using numpy comparison
    mask = max_lengths <= max_length

    # Filter using numpy boolean indexing
    new_dataset = {
        "prompt": np.array(dataset["prompt"])[mask].tolist(),
        "chosen": np.array(dataset["chosen"])[mask].tolist(),
        "rejected": np.array(dataset["rejected"])[mask].tolist()
    }

    return Dataset.from_dict(new_dataset)

def load_argilla_ds():
    # Loading it
    # train_argilla = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")
    train_argilla = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")

    # Filtering it
    train_argilla = train_argilla.filter(
        lambda r:
            r["status"] != "tie" and
            r["chosen_score"] >= 8
    )

    # Renaming to get the prompt column
    train_argilla = train_argilla.rename_column("input", "prompt")

    # Keeping only the prompt, chosen and rejected columns
    accepted_columns = set(["prompt", "chosen", "rejected"])
    all_columns = set(train_argilla.column_names)
    columns_to_remove = all_columns - accepted_columns
    train_argilla = train_argilla.remove_columns(list(columns_to_remove))

    # Formatting the dataset using the format_interaction function
    new_ds_dict = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    for r in train_argilla:
        prompt, chosen, rejected = format_interaction([r["prompt"]], r["chosen"], r["rejected"])
        new_ds_dict["prompt"].append(prompt)
        new_ds_dict["chosen"].append(chosen)
        new_ds_dict["rejected"].append(rejected)
    
    return Dataset.from_dict(new_ds_dict)

def merge_datasets(original_dpo: Dataset, perc_original: float, *datasets: Dataset):
    """
    Merges datasets with optimized performance using numpy operations.
    Args remain the same as original function.
    """
    # Convert original dataset to numpy arrays for faster operations
    final_dataset = {
        "prompt": list(original_dpo["prompt"]),
        "chosen": list(original_dpo["chosen"]),
        "rejected": list(original_dpo["rejected"])
    }

    # Concatenate all other datasets efficiently
    other_prompts = []
    other_chosen = []
    other_rejected = []
    
    for ds in datasets:
        other_prompts.extend(ds["prompt"])
        other_chosen.extend(ds["chosen"])
        other_rejected.extend(ds["rejected"])

    # Calculate samples to pick
    n_original = len(original_dpo["prompt"])
    n_samples_to_pick = min(
        int(n_original * ((1 - perc_original) / perc_original)),
        len(other_prompts)
    )

    if n_samples_to_pick > 0:
        # Use numpy's efficient random selection
        indices = np.random.choice(len(other_prompts), n_samples_to_pick, replace=False)
        
        # Extend final dataset with selected samples
        final_dataset["prompt"].extend(np.array(other_prompts)[indices])
        final_dataset["chosen"].extend(np.array(other_chosen)[indices])
        final_dataset["rejected"].extend(np.array(other_rejected)[indices])

    return Dataset.from_dict(final_dataset)
