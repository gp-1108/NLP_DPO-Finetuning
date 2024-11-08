class BaseLoader:
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