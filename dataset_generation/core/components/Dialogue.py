from .BaseComponent import BaseComponent
from .Turn import Turn
from .Chunk import Chunk
import json

class Dialogue(BaseComponent):
    """
    A class representing a dialogue composed of multiple turns.
    This class handles the creation, serialization, and management of dialogues,
    which are collections of conversation turns. Each dialogue is identified by a unique ID
    that encodes information about the document and chunks it belongs to.
    The class is based on the dialogue ID:
    - ID: A string that uniquely identifies the dialogue in the format "doc<int>_ch[<int>_<int>_<int>]" for
        each chunk that the dialogue spans.

    Attributes:
        output_file (str): Path to the output file where the dialogue will be saved.
        id (str): Unique identifier for the dialogue, following the format "doc_id_ch[chunk_ids]".
        turns (list[Turn]): List of Turn objects that make up the dialogue.
    Args:
        output_file (str, optional): Path to save the dialogue. Required if not loading from json.
        id (str, optional): Dialogue identifier. Required if not loading from json.
        turns (list[Turn], optional): List of dialogue turns. Required if not loading from json.
        json_str (str, optional): JSON string to load the dialogue from.
    Raises:
        ValueError: If neither json_str is provided nor all of (output_file, id, turns).
    Examples:
        >>> dialogue = Dialogue(output_file="output.json", id="doc1_ch[1_2_3]", turns=[turn1, turn2])
        >>> dialogue = Dialogue(json_str='{"id": "doc1_ch[1_2]", "turns": [...]}')
    """
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
    def get_id(chunk_ids: list[str]) -> str:
        """
        Generate a unique identifier from a list of chunk IDs.

        This static method creates a single ID by combining the document ID 
        and chunk numbers from a list of chunk IDs.

        Args:
            chunk_ids (list[str]): A list of chunk IDs in the format 'doc_id_chX' 
                where X is the chunk number

        Returns:
            str: A combined ID in the format 'doc_id_ch[X_Y_Z]' where X,Y,Z are 
                the chunk numbers from the input IDs
        """
        doc_id = chunk_ids[0].split("_ch")[0]
        chunk_ids = [chunk_id.split("_ch")[1] for chunk_id in chunk_ids]
        return f"{doc_id}_ch[{'_'.join(chunk_ids)}]"
    
    def get_chunk_ids(self):
        """
        Extracts and returns a list of chunk IDs from the dialogue ID.

        This method parses the dialogue ID, which contains chunk information in the format 
        "document_id_ch[chunk1_chunk2_...]", to extract individual chunk IDs. It then 
        constructs full chunk identifiers by combining the document ID with each chunk number.

        Returns:
            list: A list of complete chunk IDs, where each ID is constructed by combining
                  the document ID with the individual chunk numbers.

        Example:
            If dialogue.id = "doc1_ch[1_2_3]"
            Returns: ["doc1_ch1", "doc1_ch2", "doc1_ch3"]
        """
        chunks = self.id[self.id.index("[")+1:self.id.index("]")]
        chunk_int_ids = [int(chunk) for chunk in chunks.split("_")]
        doc_id = self.id.split("_ch")[0]
        return [Chunk.get_id(doc_id, chunk_int_id) for chunk_int_id in chunk_int_ids]
    
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