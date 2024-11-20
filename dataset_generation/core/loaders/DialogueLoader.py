import os
from .BaseLoader import BaseLoader
from ..components.Dialogue import Dialogue

class DialogueLoader(BaseLoader):
    def __init__(self, jsonl_path):
        super().__init__(jsonl_path)
    
    def load_data(self):
        """
        Load dialogues from a JSONL file and create an index mapping dialogue IDs to their positions.

        Returns:
            list: A list of Dialogue objects created from the JSONL file.
                  Returns empty list if file doesn't exist.

        Side Effects:
            Updates self.id2idx with a mapping of dialogue IDs to their index positions.
        """
        if not os.path.exists(self.jsonl_path):
            return []
        with open(self.jsonl_path, 'r') as file:
            data = [Dialogue(json_str=line) for line in file if line.strip()]
        self.id2idx = {dialogue.id: idx for idx, dialogue in enumerate(data)}
        return data
    
    def get_dialogues_by_document_id(self, document_id: str) -> list[Dialogue]:
        """
        Retrieves a list of dialogues associated with a given document ID.

        Args:
            document_id: The ID of the document to filter dialogues by.

        Returns:
            list: A list of dialogue objects where the dialogue ID contains the given document_id.
        """
        return [dialogue for dialogue in self.data if document_id in dialogue.id]
    
    def get_dialogue_by_id(self, dialogue_id: str) -> Dialogue:
        """
        Retrieves a dialogue from the dataset using its unique identifier.

        Args:
            dialogue_id: The unique identifier for the dialogue to retrieve.

        Returns:
            The dialogue object if found, None otherwise.
        """
        if dialogue_id not in self.id2idx:
            return None
        return self.data[self.id2idx[dialogue_id]]

    def load_index(self) -> set:
        """
        Loads and returns a set of all dialogue IDs from the dataset.
        Used by the __contains__ method to check if a dialogue ID exists in the dataset.

        Returns:
            set: A set containing unique dialogue IDs from the loaded data.
        """
        index = {dialogue.id for dialogue in self.data}
        return index