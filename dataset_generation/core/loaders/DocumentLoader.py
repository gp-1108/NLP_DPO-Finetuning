from ..components.Document import Document
from .BaseLoader import BaseLoader

class DocumentLoader(BaseLoader):
    def __init__(self, jsonl_path):
        super().__init__(jsonl_path)
    
    def load_data(self):
        with open(self.jsonl_path, 'r') as file:
            data = [Document(json_str=line) for line in file if line.strip()]
        return data