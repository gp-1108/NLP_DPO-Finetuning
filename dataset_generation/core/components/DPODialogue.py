from .DPOTurn import DPOTurn
from .BaseComponent import BaseComponent
from .Chunk import Chunk
import json

class DPODialogue(BaseComponent):
    def __init__(self,
                 id: str=None,
                 last_turn: DPOTurn=None,
                 json_str: str = None,
                 output_jsonl: str = None,
                ):
        if json_str:
            self.from_json_str(json_str)
        else:
            if id is None or \
               last_turn is None or \
                output_jsonl is None:
                raise Exception("DPODialogue: Missing required parameters")
            super().__init__(
                output_file=output_jsonl,
                id = id,
                last_turn=last_turn
            )

    @staticmethod
    def get_id(dialogue_id: str, rules_idx_used: list[int]):
        return f"{dialogue_id}_dpo[{'_'.join([str(idx) for idx in rules_idx_used])}]"

    @staticmethod
    def get_previous_dpo_id(dialogue_id: str):
        rules_idx_used = dialogue_id[dialogue_id.rindex("[")+1:dialogue_id.rindex("]")]
        rules_idx_used = rules_idx_used.split("_")
        rules_idx_used = [int(idx) for idx in rules_idx_used]
        rules_idx_used = rules_idx_used[:-1]
        if len(rules_idx_used) == 0:
            return None
        return DPODialogue.get_id(dialogue_id[:dialogue_id.index("_dpo")], rules_idx_used)
    
    def get_chunks_ids(self):
        chunks_int_ids = self.id[self.id.index("[")+1:self.id.index("]")]
        chunks_int_ids = chunks_int_ids.split("_")
        chunks_int_ids = [int(chunk) for chunk in chunks_int_ids]
        doc_id = self.id.split("_ch")[0]
        return [Chunk.get_id(doc_id, chunk) for chunk in chunks_int_ids]
    
    def get_doc_id(self):
        return self.id.split("_ch")[0]
    
    def to_json_str(self):
        return json.dumps({
            "id": self.id,
            "last_turn": self.last_turn.to_json_str()
        })

    def from_json_str(self, json_str: str):
        data = json.loads(json_str)
        self.id = data["id"]
        self.last_turn = DPOTurn(json_str=data["last_turn"])
    
    def __str__(self):
        string = f"Dialogue ID: {self.id}\n"
        return string
