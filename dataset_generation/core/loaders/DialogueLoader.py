import os
from .BaseLoader import BaseLoader
from ..components.Dialogue import Dialogue

class DialogueLoader(BaseLoader):
    def __init__(self, jsonl_path):
        super().__init__(jsonl_path)
    
    def load_data(self):
        if not os.path.exists(self.jsonl_path):
            return []
        with open(self.jsonl_path, 'r') as file:
            data = [Dialogue(json_str=line) for line in file if line.strip()]
        return data
    
    def get_dialogues_by_document_id(self, document_id):
        return [dialogue for dialogue in self.data if document_id in dialogue.id]
    
    def load_index(self):
        index = {dialogue.id for dialogue in self.data}
        return index