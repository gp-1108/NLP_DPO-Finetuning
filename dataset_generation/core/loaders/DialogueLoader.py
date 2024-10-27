from .BaseLoader import BaseLoader
from ..components.Dialogue import Dialogue

class DialogueLoader(BaseLoader):
    def __init__(self, jsonl_path):
        super().__init__(jsonl_path)
    
    def load_data(self):
        with open(self.jsonl_path, 'r') as file:
            data = [Dialogue(json_str=line) for line in file if line.strip()]
        return data