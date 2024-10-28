from .BaseComponent import BaseComponent
from .Chunk import Chunk
import json

class Document(BaseComponent):
    def __init__(self,
                 output_file: str=None,
                 file_name: str=None,
                 id: int=None,
                 chunks: list[Chunk]=None,
                 json_str: str = None
                ):
        if json_str:
            self.from_json_str(json_str)
        else:
            if id is None or chunks is None or output_file is None:
                raise ValueError("You either load the file from json_str or provide id, chunks, file_name, output_file")

            super().__init__(
                output_file,
                id=id,
                chunks=chunks,
                file_name=file_name
            )
    
    @staticmethod
    def get_id(id):
        return f"dc_{id}"
    
    @staticmethod
    def extract_id(id_str):
        return int(id_str.split('_')[1])
    
    def to_json_str(self):
        return json.dumps({
            "id": Document.get_id(self.id),
            "file_name": self.file_name,
            "chunks": [chunk.to_json_str() for chunk in self.chunks]
        })
    
    def from_json_str(self, json_str):
        data = json.loads(json_str)
        self.id = Document.extract_id(data["id"])
        self.file_name = data["file_name"]
        self.chunks = [Chunk(json_str=chunk) for chunk in data["chunks"]]