from .BaseSubComponent import BaseSubComponent
import json

class Chunk(BaseSubComponent):
    """
    A class representing a text chunk, inheriting from BaseSubComponent.
    This class manages text chunks with unique identifiers, providing functionality
    for creation, serialization, and string representation of text segments.
    This class here is based on the chunk id:
    - ID: A string that uniquely identifies the chunk in the format "doc<int>_ch<int>"

    Attributes:
        id (str): Unique identifier for the chunk
        text (str): The actual text content of the chunk
    Args:
        id (str, optional): The chunk identifier. Defaults to None.
        text (str, optional): The text content. Defaults to None.
        json_str (str, optional): JSON string for initialization. Defaults to None.
    Raises:
        Exception: When neither json_str is provided nor both id and text are provided.
    Methods:
        extract_ids(chunk_id): Splits chunk ID into document ID and chunk number.
        get_id(doc_id, chunk_int_id): Constructs chunk ID from document ID and chunk number.
        to_json_str(): Serializes chunk to JSON string.
        from_json_str(json_str): Initializes chunk from JSON string.
        __str__(): Returns string representation of the chunk.
    """
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
    def extract_ids(chunk_id: str) -> tuple[str, int]:
        """
        Extracts document ID and chunk ID from a combined chunk identifier.

        Args:
            chunk_id (str): Combined string identifier in format 'doc_id_chN' where N is the chunk number

        Returns:
            tuple: Contains:
                - doc_id (str): The document identifier
                - id (int): The chunk number/identifier
        """
        doc_id, id = chunk_id.split("_ch")
        return doc_id, int(id)
    
    @staticmethod
    def get_id(doc_id: str, chunk_int_id: int) -> str:
        """
        Creates a unique identifier for a chunk by combining document ID and chunk number.

        Args:
            doc_id (str): The document identifier
            chunk_int_id (int): The integer identifier of the chunk within the document

        Returns:
            str: A unique chunk identifier in the format 'doc_id_chX' where X is the chunk number
        """
        return f"{doc_id}_ch{chunk_int_id}"
    
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
        string = f"Chunk ID: {self.id}\n"
        string += f"Lenght: {len(self.text)}\n"
        string += f"Text: {self.text[:10]}...\n"
        return string