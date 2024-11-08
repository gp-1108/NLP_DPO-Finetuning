class BaseSubComponent:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_json_str(self):
        raise NotImplementedError
    
    def from_json_str(self, json_str: str):
        raise NotImplementedError
    
    def __str__(self):
        raise NotImplementedError