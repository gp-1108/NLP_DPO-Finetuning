class BaseComponent:
    """
    Base class for all components in the dataset generation pipeline.
    This class provides a common interface for components that need to be serialized to JSON
    and saved to a file in JSONL format.
    Attributes:
        output_file (str): Path to the output file where the component will be saved.
        **kwargs: Additional keyword arguments that will be set as attributes of the instance.
    Methods:
        to_json_str(): 
            Convert the component to a JSON string representation.
            Must be implemented by subclasses.
        from_json_str(json_str): 
            Create a component instance from a JSON string.
            Must be implemented by subclasses.
            Args:
                json_str (str): JSON string representation of the component.
        save():
            Append the component's JSON representation to the output file in JSONL format.
        __str__():
            Return a string representation of the component.
            Must be implemented by subclasses.
    """
    def __init__(self, output_file: str, **kwargs):
        self.output_file = output_file
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_json_str(self):
        raise NotImplementedError

    def from_json_str(self, json_str):
        raise NotImplementedError
    
    def __str__(self):
        raise NotImplementedError
    
    def save(self):
        """
        Appending the json string to the output file (JSONL format)
        """
        with open(self.output_file, 'a') as f:
            f.write(self.to_json_str())
            f.write('\n')