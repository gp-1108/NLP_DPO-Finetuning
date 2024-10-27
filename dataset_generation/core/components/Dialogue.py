from .BaseComponent import BaseComponent
from .Turn import Turn
import json

class Dialogue(BaseComponent):
    def __init__(self,
                 output_file: str=None,
                 chunk_ids: list[str] = None,
                 turns: list[Turn]=None,
                 json_str: str = None
                ):

        if json_str:
            self.from_json_str(json_str)
        else:
            if not chunk_ids or not output_file:
                raise ValueError("You either load the file from json_str or provide doc_id, chunk_ids, output_file")

            super().__init__(
                output_file,
                chunk_ids=chunk_ids,
                turns=turns
            )

    def get_id(self):
        doc_id = self.chunk_ids[0].split('_')[0]
        chunks_ids = [chunk_id.split('_')[1] for chunk_id in self.chunk_ids]
        return f"{doc_id}_ch[{'_'.join(chunks_ids)}]"
    
    @staticmethod
    def extract_ids(id_str):
        doc_id = id_str.split('_')[0]
        square_bracket_l, square_bracket_r = id_str.index('['), id_str.index(']')
        chunks = id_str[square_bracket_l+1:square_bracket_r].split('_')
        chunks_ids = [f"{doc_id}_ch_{chunk_id}" for chunk_id in chunks]
        return chunks_ids

    def to_json_str(self):
        return json.dumps({
            "id": self.get_id(),
            "turns": [turn.to_json_str() for turn in self.turns]
        })

    def from_json_str(self, json_str: str):
        data = json.loads(json_str)
        self.id = data["id"]
        self.chunk_ids = Dialogue.extract_ids(self.id)
        self.turns = [Turn(json_str=turn) for turn in data["turns"]]