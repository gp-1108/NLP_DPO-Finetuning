from .BaseSubComponent import BaseSubComponent
import json

class Turn(BaseSubComponent):
    def __init__(self,
                 role: str=None,
                 content: str=None,
                 json_str: str = None
                ):
        if json_str:
            self.from_json_str(json_str)
        else:
            if role is None or content is None:
                raise ValueError("You either load the file from json_str or provide role, content")

            super().__init__(
                role=role,
                content=content
            )
    
    def to_json_str(self):
        return json.dumps({
            "role": self.role,
            "content": self.content
        })
    
    def from_json_str(self, json_str):
        data = json.loads(json_str)
        self.role = data["role"]
        self.content = data["content"]