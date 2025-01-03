from core.loaders import DPODialogueLoader
from datasets import Dataset
import numpy as np

def format_user_input(user_input: str) -> str:
    return f"<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>"

def format_assistant_response(assistant_response: str) -> str:
    return f"<|start_header_id|>assistant<|end_header_id|>\n{assistant_response}<|eot_id|>"

def format_completion(assistant_response: str) -> str:
    return f"{assistant_response}<|eot_id|>"

def from_loader_to_pref_std_dataset(loader: DPODialogueLoader):
    dataset_dict = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    for i in range(len(loader)):
        dpo_turns = loader.get_dpo_turns_by_dialogue_id(loader[i].id)
        prompt = "<begin_of_text>"
        for i in range(0, len(dpo_turns)-1):
            prompt += format_user_input(dpo_turns[i].student_question)
            prompt += format_assistant_response(dpo_turns[i].positive_answer)
        prompt += format_user_input(dpo_turns[-1].student_question)
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
        chosen = format_completion(dpo_turns[-1].positive_answer)
        rejected = format_completion(dpo_turns[-1].negative_answer)

        dataset_dict["prompt"].append(prompt)
        dataset_dict["chosen"].append(chosen)
        dataset_dict["rejected"].append(rejected)

    return Dataset.from_dict(dataset_dict)

def load_dataset(dataset_path: str) -> Dataset:
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

    chosen_lengths = [len(tokenizer.encode(chosen)) for chosen in dataset["chosen"]]
    rejected_lengths = [len(tokenizer.encode(rejected)) for rejected in dataset["rejected"]]
    prompt_lengths = [len(tokenizer.encode(prompt)) for prompt in dataset["prompt"]]

    max_lengths = [max(chosen_lengths[i], rejected_lengths[i])+prompt_lengths[i] for i in range(len(chosen_lengths))]

    mask = [max_lengths[i] <= max_length for i in range(len(max_lengths))]

    # Creating a new dataset with the filtered samples
    new_dataset = {
        "prompt": [dataset["prompt"][i] for i in range(len(dataset["prompt"])) if mask[i]],
        "chosen": [dataset["chosen"][i] for i in range(len(dataset["chosen"])) if mask[i]],
        "rejected": [dataset["rejected"][i] for i in range(len(dataset["rejected"])) if mask[i]]
    }

    return Dataset.from_dict(new_dataset)
