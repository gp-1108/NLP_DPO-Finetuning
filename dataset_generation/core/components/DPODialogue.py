from .DPOTurn import DPOTurn
from .BaseComponent import BaseComponent
import json

class DPODialogue(BaseComponent):
    def __init__(self,
                 dialogue_id: str=None,
                 dpo_id: int=None,
                 turns: list[DPOTurn]=None,
                 json_str: str = None
                ):
        if json_str:
            self.from_json_str(json_str)
        else:
            if dialogue_id is None or \
               dpo_id is None or \
               turns is None:
                raise Exception("DPODialogue: Missing required parameters")
            super().__init__(
                dialogue_id=dialogue_id,
                dpo_id=dpo_id,
                turns=turns
            )

    def get_id(self):
        return f"{self.dialogue_id}_dpo{self.dpo_id}"
    
    def to_json_str(self):
        return json.dumps({
            "id": self.get_id(),
            "turns": [turn.to_json_str() for turn in self.turns]
        })

    def from_json_str(self, json_str: str):
        data = json.loads(json_str)
        self.dialogue_id, self.dpo_id = DPODialogue.extract_ids(data["id"])
        self.turns = [DPOTurn(json_str=json.dumps(turn)) for turn in data["turns"]]