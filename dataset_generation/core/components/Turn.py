from .BaseSubComponent import BaseSubComponent
import json

class Turn(BaseSubComponent):
    def __init__(self,
                 user: str=None,
                 assistant: str=None,
                 json_str: str = None
                ):
        if json_str:
            self.from_json_str(json_str)
        else:
            if user is None or assistant is None:
                raise ValueError("You either load the file from json_str or provide role, content")

            super().__init__(
                user=user,
                assistant=assistant
            )
    
    def to_json_str(self):
        return json.dumps({
            "user": self.user,
            "assistant": self.assistant
        })
    
    def from_json_str(self, json_str):
        data = json.loads(json_str)
        self.user = data["user"]
        self.assistant = data["assistant"]
        
    
    def __str__(self):
        string = f"User: {self.user[:10]}...\n"
        string += f"Assistant: {self.assistant[:10]}...\n"
        return string
    
    def __repr__(self):
        string = f"User: {self.user}\n"
        string += f"Assistant: {self.assistant}\n"
        return string