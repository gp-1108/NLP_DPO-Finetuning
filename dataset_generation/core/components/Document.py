from .BaseComponent import BaseComponent
from .Chunk import Chunk
import json

class Document(BaseComponent):
    """A class representing a document composed of multiple chunks.
    This class inherits from BaseComponent and provides functionality to manage document data,
    including chunks of text, file information, and document identification.
    The class is based on the document ID:
    - ID: A string that uniquely identifies the document in the format "dc_<int>", where <int> is an integer.

    Attributes:
        output_file (str): Path to the output file.
        file_name (str): Name of the document file.
        id (str): Unique identifier for the document.
        chunks (list[Chunk]): List of Chunk objects that make up the document.
    Methods:
        get_id(int_id): Generates a document ID from an integer.
        get_chunk_by_id(chunk_id): Retrieves a specific chunk by its ID.
        to_json_str(): Converts document data to JSON string.
        from_json_str(json_str): Loads document data from JSON string.
    """
    def __init__(self,
                 output_file: str=None,
                 file_name: str=None,
                 id: str=None,
                 chunks: list[Chunk]=None,
                 json_str: str = None
                ):
        if json_str:
            self.from_json_str(json_str)
        else:
            if id is None or chunks is None or output_file is None:
                raise ValueError("You either load the file from json_str or provide id, chunks, file_name, output_file")

            super().__init__(
                output_file,
                id=id,
                chunks=chunks,
                file_name=file_name
            )
    
    @staticmethod
    def get_id(int_id: int) -> str:
        return f"dc{int_id}"
    
    def get_chunk_by_id(self, chunk_id: str) -> Chunk:
        """
        Returns a specific chunk from the document based on its ID.

        Args:
            chunk_id (str): The unique identifier of the chunk to retrieve.

        Returns:
            Chunk: The chunk object with the matching ID.
                Returns None if no chunk with the specified ID is found.
        """
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None
    
    def to_json_str(self):
        return json.dumps({
            "id": self.id,
            "file_name": self.file_name,
            "chunks": [chunk.to_json_str() for chunk in self.chunks]
        })
    
    def from_json_str(self, json_str):
        data = json.loads(json_str)
        self.id = data["id"] 
        self.file_name = data["file_name"]
        self.chunks = [Chunk(json_str=chunk) for chunk in data["chunks"]]
    
    def __str__(self):
        string = f"Document ID: {self.id}\n"
        string += f"File Name: {self.file_name}\n"
        string += f"Chunks: {[chunk.id for chunk in self.chunks]}\n"
        return string