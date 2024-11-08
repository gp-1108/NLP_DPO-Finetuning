from ..components.DPODialogue import DPODialogue
from .BaseLoader import BaseLoader

class DPODialogueLoader(BaseLoader):
    def __init__(self, jsonl_path):
        super().__init__(jsonl_path)
    
    def load_data(self):
        with open(self.jsonl_path, 'r') as file:
            data = [DPODialogue(json_str=line) for line in file if line.strip()]
        return data
    
    def load_index(self):
        index = {dialogue.id for dialogue in self.data}
        return index