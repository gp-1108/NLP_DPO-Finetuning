from .BaseComponent import BaseComponent
from .Turn import Turn
import json

class Dialogue(BaseComponent):
    def __init__(self,
                 output_file: str=None,
                 id: str=None,
                 turns: list[Turn]=None,
                 json_str: str = None
                ):

        if json_str:
            self.from_json_str(json_str)
        else:
            if id is None or output_file is None or turns is None:
                raise ValueError("You either load the file from json_str or provide doc_id, chunk_ids, output_file")

            super().__init__(
                output_file,
                id=id,
                turns=turns
            )
    
    @staticmethod
    def get_id(chunk_ids):
        doc_id = chunk_ids[0].split("_ch")[0]
        chunk_ids = [chunk_id.split("_ch")[1] for chunk_id in chunk_ids]
        return f"{doc_id}_ch[{'_'.join(chunk_ids)}]"
    
    def to_json_str(self):
        return json.dumps({
            "id": self.id,
            "turns": [turn.to_json_str() for turn in self.turns]
        })

    def from_json_str(self, json_str: str):
        data = json.loads(json_str)
        self.id = data["id"]
        self.turns = [Turn(json_str=turn) for turn in data["turns"]]
    
    def __str__(self):
       string = f"Dialogue: {self.id}\n"
       string += f"Turns: {len(self.turns)}\n"
       return string