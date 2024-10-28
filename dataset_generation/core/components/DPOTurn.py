from .BaseSubComponent import BaseSubComponent
import json

class DPOTurn(BaseSubComponent):
    def __init__(self,
                 role: str=None,
                 positive_sample: bool=None,
                 negative_sample: bool=None,
                 rule_used: int=None,
                 json_str: str = None
                ):
        if json_str:
            self.from_json_str(json_str)
        else:
            if role is None or \
                positive_sample is None or \
                negative_sample is None or \
                rule_used is None:
                raise Exception("DPOTurn: Missing required parameters")
            if positive_sample and negative_sample:
                raise Exception("DPOTurn: positive_sample and negative_sample can't be both True")
            super().__init__(
                role=role,
                positive_sample=positive_sample,
                negative_sample=negative_sample,
                rule_used=rule_used
            )
    
    def to_json_str(self):
        return json.dumps({
            "role": self.role,
            "positive_sample": self.positive_sample,
            "negative_sample": self.negative_sample,
            "rule_used": self.rule_used
        })
    
    def from_json_str(self, json_str: str):
        data = json.loads(json_str)
        self.role = data["role"]
        self.positive_sample = data["positive_sample"]
        self.negative_sample = data["negative_sample"]
        self.rule_used = data["rule_used"]