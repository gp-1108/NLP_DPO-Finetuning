from .BaseSubComponent import BaseSubComponent
import json

class DPOTurn(BaseSubComponent):
    def __init__(self,
                 student_question: str=None,
                 positive_answer: str=None,
                 negative_answer: str=None,
                 rule_used: int=None,
                 json_str: str = None
                ):
        if json_str:
            self.from_json_str(json_str)
        else:
            if student_question is None or \
                positive_answer is None or \
                negative_answer is None or \
                rule_used is None:
                raise Exception("DPOTurn: Missing required parameters")
            super().__init__(
                student_question=student_question,
                positive_answer=positive_answer,
                negative_answer=negative_answer,
                rule_used=rule_used
            )
    
    def to_json_str(self):
        return json.dumps({
            "positive_answer": self.positive_answer,
            "negative_answer": self.negative_answer,
            "rule_used": self.rule_used,
            "student_question": self.student_question
        })
    
    def from_json_str(self, json_str: str):
        data = json.loads(json_str)
        self.positive_answer = data["positive_answer"]
        self.negative_answer = data["negative_answer"]
        self.rule_used = data["rule_used"]
        self.student_question = data["student_question"]
    
    def __str__(self):
        string = f"Student: {self.student_question[:10]}...\n"
        string += f"Rule Used: {self.rule_used}\n"
        string += f"Positive answer: {self.positive_answer[:10]}...\n"
        string += f"Negative answer: {self.negative_answer[:10]}...\n"
        return string
    
    def __repr__(self):
        string = f"Rule Used: {self.rule_used}\n"
        string += f"Student: {self.student_question}\n"
        string += f"Positive answer: {self.positive_answer}\n"
        string += f"Negative answer: {self.negative_answer}\n"
        return string