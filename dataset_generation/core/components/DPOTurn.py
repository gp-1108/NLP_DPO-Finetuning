from .BaseSubComponent import BaseSubComponent
import json

class DPOTurn(BaseSubComponent):
    """A class representing a DPO (Direct Preference Optimization) conversational turn.
    This class inherits from BaseSubComponent and stores information about a single turn
    in a conversational exchange, including the student's question and two possible answers
    (positive and negative) along with the rule that was used.
    Attributes:
        student_question (str): The question asked by the student
        positive_answer (str): The preferred/positive response to the question
        negative_answer (str): The less preferred/negative response to the question  
        rule_used (int): Identifier for the rule applied in this turn
    Methods:
        to_json_str(): Converts the turn data to a JSON string
        from_json_str(json_str): Populates the turn data from a JSON string
        __str__(): Returns a shortened string representation of the turn
        __repr__(): Returns a complete string representation of the turn
    """
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