import os
from ..components.Document import Document
from .BaseLoader import BaseLoader

class DocumentLoader(BaseLoader):
    def __init__(self, jsonl_path: str):
        super().__init__(jsonl_path)
    
    def load_data(self):
        if not os.path.exists(self.jsonl_path):
            return []
        with open(self.jsonl_path, 'r') as file:
            data = [Document(json_str=line) for line in file if line.strip()]
        self.strid2idx = {doc.id: idx for idx, doc in enumerate(data)}
        return data
    
    def get_document_by_id(self, doc_id):
        return self.data[self.strid2idx[doc_id]]
    
    def load_index(self):
        index = {document.id for document in self.data}
        return index