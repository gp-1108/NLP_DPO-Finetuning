from .BaseSubComponent import BaseSubComponent
import json

class Chunk(BaseSubComponent):
    def __init__(self,
                 id: str=None,
                 text: str=None,
                 json_str: str = None
                ):
        if json_str:
            self.from_json_str(json_str)
        else:
            if  id is None or \
                text is None:
                print(not id, not text)
                raise Exception("Chunk: Missing required parameters")
            super().__init__(
                id=id,
                text=text
            )
    
    @staticmethod
    def extract_ids(chunk_id: str):
        doc_id, id = chunk_id.split("_ch_")
        return doc_id, int(id)
    
    @staticmethod
    def get_id(doc_id, chunk_int_id):
        return f"{doc_id}_ch_{chunk_int_id}"
    
    def to_json_str(self):
        return json.dumps({
            "id": self.id,
            "text": self.text
        })
    
    def from_json_str(self, json_str: str):
        data = json.loads(json_str)
        self.id = data["id"]
        self.text = data["text"]
    
    def __str__(self):
        return f"Chunk ID: {self.id}\nText: {self.text}\n"