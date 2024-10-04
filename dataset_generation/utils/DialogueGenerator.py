from openai import OpenAI
from pydantic import BaseModel
import os

class InteractionSchema(BaseModel):
    student: str
    tutor: str

class DialogueSchema(BaseModel):
    interactions: list[InteractionSchema]

class DialogueGenerator:
    def __init__(self, jsons_path: str, prompt_path: str, output_dir: str, model: str = "gpt-4o"):
        """
        This class is a wrapper around the OpenAI API. It is meant to be used for creating dialogues
        Each json file is a list of strings where each string contains a somewhat coherent piece of text
        extracted from a pdf file. The prompt is a text file containing the prompt for the OpenAI API
        with the <SOURCE_TEXT> token to be replaced by the text extracted from the pdf file.
        As an output the class will generate a dialogue between a student and a tutor based on the text
        by querying the OpenAI API.
        """
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = model
        self.json_files = DialogueGenerator._load_jsons(jsons_path)
        self.output_dir = output_dir
        self.prompt = open(prompt_path, "r").read() # The prompt with the <SOURCE_TEXT> token to be replaced
    
    def _query_openai(self, source_text: str) -> list[dict]:
        """
        This function will query the OpenAI API with the source text.

        Args:
            source_text (str): The source text to be used in the prompt.

        Returns:
            list[dict]: A list of dictionaries with the format {"student": str, "tutor": str}
        """
        prompt = self._generate_prompt(source_text)
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format=DialogueSchema
        )
        # Parsing back to python object
        completion = completion.to_dict()
        dialogue = completion["choices"][0]["message"]["parsed"]["interactions"]
        return dialogue
    
    def _generate_prompt(self, source_text: str) -> str:
        """
        This function will generate the prompt for the OpenAI API.

        Args:
            json_path (str): The path to the json file.

        Returns:
            str: The prompt for the OpenAI API.
        """
        return self.prompt.replace("<SOURCE_TEXT>", source_text)
    
    @staticmethod
    def _load_jsons(jsons_path: str) -> list[str]:
        """
        This function will load in memory all paths to the json files.

        Args:
            jsons_path (str): The path to the json files.

        Returns:
            list[str]: A list of paths to the json files.
        """
        json_files = []
        for root, _, files in os.walk(jsons_path):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))
        return json_files
        
        