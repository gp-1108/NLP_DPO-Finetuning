from openai import OpenAI
from pydantic import BaseModel
import os
import json
from tqdm import tqdm

class InteractionSchema(BaseModel):
    student_question: str
    tutor_response: str

class DialogueSchema(BaseModel):
    dialogue: list[InteractionSchema]

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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.prompt = open(prompt_path, "r").read() # The prompt with the <SOURCE_TEXT> token to be replaced
    
    def generate_dialogues(self):
        """
        This function will generate dialogues for all json files.
        """
        for json_file in tqdm(self.json_files, desc="Processing JSON files"):
            text_chunks = self._load_text_chunks(json_file)
            source_texts = self._define_source_texts(text_chunks)
            dialogues = []
            for source_text in source_texts:
                dialogue = self.generate_dialogue(source_text)
                dialogues.append(dialogue)
            new_json_path = os.path.join(self.output_dir, os.path.basename(json_file))
            self._save_dialogues(dialogues, new_json_path)

    def _define_source_texts(self, text_chunks: list[str]) -> list[str]:
        """
        This function will define the source texts for the dialogues given the original text chunks
        of the pdf file (extracted with the TextExtractor class).

        Args:
            text_chunks (list[str]): A list of text chunks.

        Returns:
            list[str]: A list of source texts, combining the text chunks.
        """
        if not text_chunks:
            return []
        # [TODO] Carefully think about this function as it is crucial for the quality of the dialogues

        
        # 1. Merge chunks until the length is around 5K chars
        # source_texts = []
        # source_text = text_chunks[0]
        # for chunk in text_chunks[1:]:
        #     if len(source_text) + len(chunk) > 5000:
        #         source_texts.append(source_text)
        #         source_text = chunk
        #     else:
        #         source_text += chunk
        #         source_texts.append(source_text)

        # 2. Merge chunks with length around 3K but do overlaps from left and right of 1K
        source_texts = []
        start, end = 0, 0
        length = 0
        while end < len(text_chunks):
            # First let's check that we have a text longer than 5K
            length += len(text_chunks[end])
            if length > 5000:
                # Now we add the the overlapping chunks from the left and right
                if start > 0:
                    source_text = text_chunks[start-1][-1000:]
                source_text = text_chunks[start]
                for i in range(start+1, end+1):
                    source_text += text_chunks[i]
                if end < len(text_chunks) - 1:
                    source_text += text_chunks[end+1][-1000:]
                source_texts.append(source_text)
                start, end = end+1, end+1
                length = 0
            else:
                end += 1
        return source_texts
                
        
    def _load_text_chunks(self, json_path: str) -> list[str]:
        """
        This function will load the text chunks from a json file.

        Args:
            json_path (str): The path to the json file.

        Returns:
            list[str]: A list of text chunks.
        """
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def _save_dialogues(self, dialogue: list[list[dict]], json_path: str):
        """
        This function will save the dialogues in a json file.

        Args:
            dialogue (list[dict]): The dialogues to be saved.
            json_path (str): The path to the json file.
        """
        with open(json_path, 'w') as f:
            json.dump(dialogue, f, indent=2)


    def generate_dialogue(self, source_text: str) -> list[dict]:
        """
        This function will generate a dialogue based on the source text.

        Args:
            source_text (str): The source text to be used in the prompt.

        Returns:
            list[dict]: A list of dictionaries with the format {"student_question": str, "tutor_response": str}
        """
        return self._query_openai(source_text)
    
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
        dialogue = completion["choices"][0]["message"]["parsed"]["dialogue"]
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