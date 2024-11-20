import os
from ..components.Document import Document
from .BaseLoader import BaseLoader

class DocumentLoader(BaseLoader):
    def __init__(self, jsonl_path: str):
        super().__init__(jsonl_path)
    
    def load_data(self) -> list[Document]:
        """
        Load and process data from a JSONL file into Document objects.

        This method reads a JSONL file specified by self.jsonl_path and creates Document 
        objects from each non-empty line. It also builds an index mapping document IDs
        to their positions in the resulting list.

        Returns:
            list[Document]: A list of Document objects created from the JSONL file data.
                            Returns empty list if file does not exist.

        Note:
            - Skips empty lines in the input file
            - Updates self.strid2idx with {document_id: index} mapping
        """
        if not os.path.exists(self.jsonl_path):
            return []
        with open(self.jsonl_path, 'r') as file:
            data = [Document(json_str=line) for line in file if line.strip()]
        self.strid2idx = {doc.id: idx for idx, doc in enumerate(data)}
        return data
    
    def get_document_by_id(self, doc_id: str) -> Document:
        """
        Retrieves a document from the dataset using its unique identifier.

        Args:
            doc_id (str): The unique identifier of the document to retrieve.

        Returns:
            dict: The document corresponding to the given ID. The structure depends on the dataset format.

        Raises:
            KeyError: If doc_id is not found in the dataset.
        """
        return self.data[self.strid2idx[doc_id]]
    
    def load_index(self):
        """
        Returns a set containing all document IDs from the loaded data.
        Used by the __contains__ method to check if a document ID exists in the dataset.

        Returns:
            set: A set of document IDs extracted from the data collection.
        """
        index = {document.id for document in self.data}
        return index