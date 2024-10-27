class BaseComponent:
    def __init__(self, output_file: str, **kwargs):
        self.output_file = output_file
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_json_str(self):
        raise NotImplementedError
    
    def save(self):
        """
        Appending the json string to the output file (JSONL format)
        """
        if not self.stand_alone:
            raise ValueError("This method should only be called on stand alone components, which are meant to be saved to disk by themselves.")
        with open(self.output_file, 'a') as f:
            f.write(self.to_json_str())
            f.write('\n')