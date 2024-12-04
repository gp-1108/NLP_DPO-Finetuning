from core.loaders import DPODialogueLoader
from datasets import Dataset

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
