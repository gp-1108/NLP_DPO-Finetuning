class BaseLoader:
    """Base class for dataset loaders.
    This class provides a foundation for implementing dataset loaders that read from JSONL files.
    It implements basic dictionary-like behavior with key lookups and length queries.
    Args:
        jsonl_path (str): Path to the JSONL file containing the dataset.
    Attributes:
        jsonl_path (str): Path to the JSONL file.
        data: The loaded dataset (format depends on implementation).
        index: Index structure for the dataset (format depends on implementation).
    Methods:
        load_data(): Abstract method to load the dataset from the JSONL file.
        load_index(): Abstract method to create an index for the dataset.
        __contains__(key): Check if a key exists in the index.
        __len__(): Get the number of items in the dataset.
        __getitem__(key): Get an item from the dataset by key.
    Raises:
        NotImplementedError: When load_data() or load_index() are not implemented by child class.
    """
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path
        self.data = self.load_data()
        self.index = self.load_index()
    
    def load_data(self):
        raise NotImplementedError

    def load_index(self):
        raise NotImplementedError
    
    def __contains__(self, key):
        return key in self.index

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        return self.data[key]