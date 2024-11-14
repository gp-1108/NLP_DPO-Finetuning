from openai import OpenAI
from pydantic import BaseModel
import os
import json
from tqdm import tqdm
from ..loaders import DocumentLoader
from ..components import Chunk, Dialogue, Turn
from ..loaders import DialogueLoader

class InteractionSchema(BaseModel):
    student_question: str
    tutor_response: str

class DialogueSchema(BaseModel):
    dialogue: list[InteractionSchema]

class DialogueGenerator:
    def __init__(self,
                 jsonl_file: str,
                 output_jsonl: str,
                 prompt_path: str,
                 model: str = "gpt-4o"
                ):
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
        self.docs = DocumentLoader(jsonl_file)
        self.output_jsonl = output_jsonl
        self.already_processed = DialogueLoader(output_jsonl)
        self.prompt = open(prompt_path, "r").read() # The prompt with the <SOURCE_TEXT> token to be replaced
    
    def generate_all(self):
        for doc in tqdm(self.docs):
            source_texts = self._define_source_texts(doc.chunks)
            for chunk_ids, source_text in source_texts:
                dialogue_id = Dialogue.get_id(chunk_ids)
                if dialogue_id in self.already_processed:
                    print(f"Dialogue with ID {dialogue_id} already processed.")
                    continue
                dialogue_list = self._query_openai(source_text)
                dialogue = self.create_dialogue(dialogue_list, dialogue_id)
                dialogue.save()

    def create_dialogue(self, dialogue: list[dict], dialogue_id: str) -> Dialogue:
        """
        This function will create a Dialogue instance from the dialogue list.

        Args:
            dialogue (list[dict]): A list of dictionaries with the format {"student": str, "tutor": str}

        Returns:
            Dialogue: A Dialogue instance.
        """
        turns = []
        for interaction in dialogue:
            turn = Turn(user=interaction["student_question"], assistant=interaction["tutor_response"])
            turns.append(turn)
        return Dialogue(output_file=self.output_jsonl, id=dialogue_id, turns=turns)
            
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
    
    def _define_source_texts(self, text_chunks: list[Chunk]) -> list[tuple[list[str], str]]:
        """
        This function will define the source texts for the dialogues given the original text chunks.

        Args:
            text_chunks (list[Chunk]): A list of Chunk instances.

        Returns:
            list[tuple[list[str], str]]: A list of tuples where each tuple contains a list of chunk IDs and the merged text content.
        """
        if not text_chunks:
            return []

        source_texts = []
        start, end = 0, 0
        length = 0

        while end < len(text_chunks):
            length += len(text_chunks[end].text)
            
            # Check if the accumulated length reaches around 5K characters
            if length >= 5000:
                # Include 1K-character overlaps from the previous and next chunks, if available
                chunk_ids = [text_chunks[i].id for i in range(start, end + 1)]
                merged_text = ""

                if start > 0:
                    merged_text += text_chunks[start - 1].text[-1000:]  # Left overlap
                
                for i in range(start, end + 1):
                    merged_text += text_chunks[i].text

                if end < len(text_chunks) - 1:
                    merged_text += text_chunks[end + 1].text[:1000]  # Right overlap

                source_texts.append((chunk_ids, merged_text))
                
                # Move to the next segment
                start = end + 1
                end = start
                length = 0
            else:
                end += 1

        # Handle any remaining chunks after the loop
        if start < len(text_chunks):
            chunk_ids = [text_chunks[i].id for i in range(start, len(text_chunks))]
            merged_text = ""

            if start > 0:
                merged_text += text_chunks[start - 1].text[-1000:]

            for i in range(start, len(text_chunks)):
                merged_text += text_chunks[i].text

            source_texts.append((chunk_ids, merged_text))

        return source_texts
