from pydantic import BaseModel
from openai import OpenAI
import os
import json
from tqdm import tqdm

class ResponseSchema(BaseModel):
    adapted_student_response: str
    good_tutor_response: str
    bad_tutor_response: str

class DPOGenerator:
    def __init__(
            self,
            jsons_path: str,
            prompt_path: str,
            output_dir: str,
            model: str = "gpt-4o"
        ):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = model
        self.json_files = DPOGenerator._load_jsons(jsons_path)
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.prompt = open(prompt_path, "r").read()
    
    def generate_dialogue_trees(self):
        """
        Generates dialogue trees for each JSON file in the json_files list.

        This method iterates over all JSON files, loads the dialogue data from each file,
        generates a dialogue tree from the loaded dialogue, and then saves the generated
        dialogue tree back to the corresponding JSON file.

        Returns:
            None
        """
        for json_file in tqdm(self.json_files, desc="Generating dialogue trees"):
            dialogues = self._load_dialogues(json_file)
            dpo_dialogues = []
            for dialogue in dialogues:
                dialogue_tree = self.generate_dialogue_tree(dialogue)
                dpo_dialogues.append(dialogue_tree)
            self._save_dialogues_tree(dpo_dialogues, json_file)

    def generate_dialogue_tree(self, dialogue: list[dict]) -> list[str]:
        """
        This function will generate a dialogue tree based on the given dialogue.

        Args:
            dialogue (list[dict]): A list of student-tutor interactions. Each interaction is a dictionary
            with the keys "student_question" and "tutor_response".

        Returns:
            list[dict]: A list of objects with the format {"student": str, "good_tutor": str, "bad_tutor": str}
        """
        generated_tree = []
        last_tut_response = "// the conversation has just started, no tutor’s response"
        for interaction in dialogue:
            student_question = interaction["student_question"]
            tutor_response = interaction["tutor_response"]
            adapted_student_response, good_tutor_response, bad_tutor_response = self._query_openai(
                last_tut_response, student_question, tutor_response
            )
            generated_tree.append({
                "adapted_student_response": adapted_student_response,
                "good_tutor_response": good_tutor_response,
                "bad_tutor_response": bad_tutor_response
            })
            last_tut_response = good_tutor_response
        return generated_tree
    
    def _load_dialogues(self, json_file: str) -> list[dict]:
        """
        This function will load all the dialogues from the given json file.

        Args:
            json_file (str): The path to the json file.

        Returns:
            list[list[dict]]: A list of list of student-tutor interactions. Each interaction is a dictionary
            with the keys "student_question" and "tutor_response".
        """
        with open(json_file, "r") as f:
            dialogue = json.load(f)
        return dialogue
    
    def _save_dialogues_tree(self, dialogues_tree: list[list[str]], json_file: str):
        """
        This function will save the generated dialogue tree to a JSON file.
        The format will be the same as the input obj file.

        Args:
            dialogue_tree (list[list[str]]): The generated dialogue trees
            json_file (str): The path to the json file.
        """
        output_file = os.path.join(self.output_dir, os.path.basename(json_file))
        with open(output_file, "w") as f:
            json.dump(dialogues_tree, f, indent=2)
    
    def _query_openai(self, last_tut_response: str, student_question: str, tutor_response: str) -> str:
        prompt = self._generate_prompt(last_tut_response, student_question, tutor_response)
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format=ResponseSchema
        )
        completion = completion.to_dict()
        answer = completion["choices"][0]["message"]["parsed"]
        adapted_student_response = answer["adapted_student_response"]
        good_tutor_response = answer["good_tutor_response"]
        bad_tutor_response = answer["bad_tutor_response"]
        return adapted_student_response, good_tutor_response, bad_tutor_response
    
    def _generate_prompt(self, last_tut_response: str, student_question: str, tutor_response: str) -> str:
        """
        This function will generate the prompt for the OpenAI API.

        Args:
            last_tut_response (str): The last tutor response.
            student_question (str): The student question.
            tutor_response (str): The tutor response.

        Returns:
            str: The generated prompt.
        """
        prompt = self.prompt.replace("<LAST_TUTOR_RESPONSE>", last_tut_response)
        prompt = prompt.replace("<STUDENT_QUESTION>", student_question)
        prompt = prompt.replace("<TUTOR_RESPONSE>", tutor_response)
        return prompt
    
    @staticmethod
    def _load_jsons(jsons_path: str) -> list[str]:
        """
        This function will load all the json files paths in the given directory.

        Args:
            jsons_path (str): The path to the directory containing the json files.

        Returns:
            list[str]: A list of paths to the json files.
        """
        files = []
        for root, _, filenames in os.walk(jsons_path):
            for filename in filenames:
                if filename.endswith(".json"):
                    files.append(os.path.join(root, filename))
        return files