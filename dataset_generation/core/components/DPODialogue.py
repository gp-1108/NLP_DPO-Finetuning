from .DPOTurn import DPOTurn
from .BaseComponent import BaseComponent
from .Chunk import Chunk
import json

class DPODialogue(BaseComponent):
    """
    A class representing a Direct Preference Optimization (DPO) dialogue component.
    This class manages dialogue data with DPO-specific functionality, including ID management,
    dialogue history tracking, and JSON serialization/deserialization.
    Parameters
    ----------
    id : str, optional
        Unique identifier for the dialogue. Should follow the format 'dc<int>_ch[<int>_<int>...]_dpo[<int>_<int>...]'
    last_turn : DPOTurn, optional
        The last turn/interaction in the dialogue
    json_str : str, optional
        JSON string representation of a DPODialogue for deserialization
    output_jsonl : str, optional
        Path to the output JSONL file where dialogue data will be saved
    Attributes
    ----------
    id : str
        Unique identifier for the dialogue
    last_turn : DPOTurn
        The last turn/interaction in the dialogue
    Methods
    get_id(dialogue_id: str, rules_idx_used: list[int]) -> str
        Static method to generate a DPO dialogue ID from base dialogue ID and rule indices
    get_previous_dpo_id(dialogue_id: str) -> str
        Static method to get the ID of the previous dialogue in the DPO chain
    get_chunks_ids() -> list[str]
        Get list of chunk IDs associated with this dialogue
    get_doc_id() -> str
        Get the document ID from the dialogue ID
    to_json_str() -> str
        Serialize the dialogue to a JSON string
    from_json_str(json_str: str)
        Deserialize a JSON string into a DPODialogue object
    """
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
    def get_previous_dpo_id(dialogue_id: str) -> str:
        """
        Get the ID of the previous DPO dialogue in the history chain.

        The function extracts and processes the rule indices from the current dialogue ID,
        removes the last rule index, and generates a new ID for the previous dialogue.

        Args:
            dialogue_id (str): The current dialogue ID containing rule indices in the format
                              "base_id_dpo[idx1_idx2_...]"

        Returns:
            str: The ID of the previous dialogue in the DPO chain.
                 Returns None if there are no previous dialogues (i.e., this is the first in the chain).

        Example:
            >>> get_previous_dpo_id("dialogue1_dpo[1_2_3]")
            "dialogue1_dpo[1_2]"
            >>> get_previous_dpo_id("dialogue1_dpo[1]")
            None
        """
        rules_idx_used = dialogue_id[dialogue_id.rindex("[")+1:dialogue_id.rindex("]")]
        rules_idx_used = rules_idx_used.split("_")
        rules_idx_used = [int(idx) for idx in rules_idx_used]
        rules_idx_used = rules_idx_used[:-1]
        if len(rules_idx_used) == 0:
            return None
        return DPODialogue.get_id(dialogue_id[:dialogue_id.index("_dpo")], rules_idx_used)
    
    def get_chunks_ids(self) -> list[str]:
        """
        Extract individual chunk IDs from a DPO dialogue ID.

        The method parses the dialogue ID which contains chunk numbers in the format 
        'document_id_ch[chunk1_chunk2_...]' and returns a list of full chunk IDs.

        Returns
        -------
        list
            A list of chunk IDs in the format 'document_id_chX' where X is the chunk number.

        Example
        -------
        For dialogue ID 'doc1_ch[1_2_3]', returns ['doc1_ch1', 'doc1_ch2', 'doc1_ch3']
        """
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
