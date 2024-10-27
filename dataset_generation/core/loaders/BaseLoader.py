class BaseLoader:
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path
        self.data = self.load_data()
    
    def load_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]