from .BaseSubComponent import BaseSubComponent
import json

class Chunk(BaseSubComponent):
    def __init__(self,
                 id: int=None,
                 doc_id: str=None,
                 text: str=None,
                 json_str: str = None
                ):
        if json_str:
            self.from_json_str(json_str)
        else:
            if  id is None or \
                doc_id is None or \
                text is None:
                print(not id, not doc_id, not text)
                raise Exception("Chunk: Missing required parameters")
            super().__init__(
                id=id,
                doc_id=doc_id,
                text=text
            )
    
    @staticmethod
    def extract_ids(chunk_id: str):
        doc_id, id = chunk_id.split("_ch_")
        return doc_id, int(id)
    
    def get_id(self):
        return f"{self.doc_id}_ch_{self.id}"
    
    def to_json_str(self):
        return json.dumps({
            "id": self.get_id(),
            "text": self.text
        })
    
    def from_json_str(self, json_str: str):
        data = json.loads(json_str)
        self.id = data["id"]
        self.doc_id, self.id = Chunk.extract_ids(self.id)
        self.text = data["text"]