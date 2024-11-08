from .DPOTurn import DPOTurn
from .BaseComponent import BaseComponent
import json

class DPODialogue(BaseComponent):
    def __init__(self,
                 id: str=None,
                 turns: list[DPOTurn]=None,
                 json_str: str = None
                ):
        if json_str:
            self.from_json_str(json_str)
        else:
            if id is None or \
               turns is None:
                raise Exception("DPODialogue: Missing required parameters")
            super().__init__(
                id = id,
                turns=turns
            )

    @staticmethod
    def get_id(dialogue_id: str, dpo_int_id: int):
        return f"{dialogue_id}_dpo{dpo_int_id}"
    
    @staticmethod
    def extract_ids(dialogue_id: str):
        doc_id, id = dialogue_id.split("_dpo")
        return doc_id, int(id)
    
    def to_json_str(self):
        return json.dumps({
            "id": self.get_id(),
            "turns": [turn.to_json_str() for turn in self.turns]
        })

    def from_json_str(self, json_str: str):
        data = json.loads(json_str)
        self.dialogue_id, self.dpo_id = DPODialogue.extract_ids(data["id"])
        self.turns = [DPOTurn(json_str=json.dumps(turn)) for turn in data["turns"]]
    
    def __str__(self):
        string = f"Dialogue ID: {self.dialogue_id}\n"
