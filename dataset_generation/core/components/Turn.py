from .BaseSubComponent import BaseSubComponent
import json

class Turn(BaseSubComponent):
    """A class representing a turn in a conversation between a user and an assistant.
    This class inherits from BaseSubComponent and handles the storage and serialization
    of a conversation turn, which consists of a user message and an assistant response.
    Args:
        user (str, optional): The user's message in the conversation turn.
        assistant (str, optional): The assistant's response in the conversation turn.
        json_str (str, optional): A JSON string representation of a turn to load from.
    Raises:
        ValueError: If neither json_str is provided nor both user and assistant are provided.
    Attributes:
        user (str): The user's message in the conversation turn.
        assistant (str): The assistant's response in the conversation turn.
    Methods:
        to_json_str(): Converts the turn to a JSON string representation.
        from_json_str(json_str): Loads the turn from a JSON string representation.
        __str__(): Returns a truncated string representation of the turn.
        __repr__(): Returns a complete string representation of the turn.
    """
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