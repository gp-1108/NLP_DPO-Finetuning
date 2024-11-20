from openai import OpenAI
from pydantic import BaseModel
import os
from tqdm import tqdm
from ..loaders import DocumentLoader, DialogueLoader
from ..components import Chunk, Dialogue, Turn, Document

class InteractionSchema(BaseModel):
    student_question: str
    tutor_response: str

class DialogueSchema(BaseModel):
    dialogue: list[InteractionSchema]

class DialogueGenerator:
    """
    A class for generating dialogues between a student and a tutor using OpenAI's API.
    This class processes text documents by breaking them into chunks, sends them to OpenAI's API
    with a specified prompt, and generates educational dialogues based on the content. The dialogues
    are saved in a JSONL format.
    Attributes:
        client (OpenAI): OpenAI client instance for API calls
        model (str): The OpenAI model to use (default: "gpt-4")
        docs (DocumentLoader): Document loader instance for reading input files
        output_jsonl (str): Path to the output JSONL file
        already_processed (DialogueLoader): Loader for tracking processed dialogues
        prompt (str): Template prompt with <SOURCE_TEXT> placeholder
        jsonl_file (str): Path to input JSONL file containing source texts
        output_jsonl (str): Path to output JSONL file for storing generated dialogues
        prompt_path (str): Path to prompt template file
        model (str, optional): OpenAI model name. Defaults to "gpt-4"
        - Requires OpenAI API key set in environment variables
        - Input JSONL should contain coherent text chunks from PDF files
        - Prompt file should contain <SOURCE_TEXT> token for replacement
        - Generates structured dialogues between student and tutor
        - Handles text in chunks of ~5000 characters with 1000-character overlaps
    """
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
    
    def generate_all(self) -> None:
        """
        Generate dialogues for all documents in the dataset.

        Iterates through all documents in self.docs and attempts to generate a dialogue
        for each one. If an error occurs during generation for a specific document,
        the error is printed and processing continues with the next document.

        Raises:
            No direct exceptions, but may print errors from generate_single_dialogue()
        """
        for doc in tqdm(self.docs):
            try:
                self.generate_single_dialogue(doc)
            except Exception as e:
                print(f"Error while processing document {doc.id}: {e}")
    
    def generate_single_dialogue(self, doc: Document) -> None:
        """
        Generate a dialogue from a given document.

        This method processes a document by breaking it into chunks, generating dialogues
        from these chunks using OpenAI's API, and saving the resulting dialogues.

        Args:
            doc (Document): The document object containing text chunks to be processed.

        Returns:
            None

        Note:
            - The method skips already processed dialogues based on their IDs
            - Dialogues are generated using OpenAI's API
            - Generated dialogues are saved to storage
        """
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
        Organizes text chunks into overlapping segments of approximately 5000 characters each.
        This method combines consecutive text chunks into larger segments while maintaining
        1000-character overlaps between adjacent segments to preserve context. Each segment
        consists of complete chunks that sum up to roughly 5000 characters, plus overlapping
        portions from neighboring chunks.
        Args:
            text_chunks (list[Chunk]): A list of Chunk objects, each containing text and an ID.
        Returns:
            list[tuple[list[str], str]]: A list of tuples where each tuple contains:
                - list[str]: IDs of the chunks included in the segment
                - str: The merged text including overlapping portions
        Notes:
            - Each segment aims to be around 5000 characters
            - Includes up to 1000 characters overlap from previous and next chunks
            - Last segment may be shorter than 5000 characters
            - Returns empty list if input is empty
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
