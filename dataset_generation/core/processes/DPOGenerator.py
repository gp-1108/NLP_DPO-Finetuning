from ..components import PedagogicalRules, DPODialogue, DPOTurn, Turn, Dialogue
from ..loaders import DPODialogueLoader, DialogueLoader
from openai import OpenAI
from pydantic import BaseModel
import os
import random
from tqdm import tqdm

class UseRuleSchema(BaseModel):
    rule_fit_score: int

class GoodAnswerSchema(BaseModel):
    adapted_response: str
    tutor_response: str

class BadAnswerSchema(BaseModel):
    not_following_tutor: str

class DPOGenerator:
    K = 3

    def __init__(self,
                 jsonl_file: str,
                 output_jsonl: str,
                 rules_txt_path: str,
                 good_answer_prompt_path: str,
                 bad_answer_prompt_path: str,
                 apply_rule_prompt_path: str,
                 model: str = "gpt-4o"):
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.dialogues = DialogueLoader(jsonl_file)
        self.output_jsonl = output_jsonl
        self.already_processed = DPODialogueLoader(output_jsonl)
        self.rules = PedagogicalRules(rules_txt_path)
        self.good_answer_prompt = open(good_answer_prompt_path, "r").read()
        self.bad_answer_prompt = open(bad_answer_prompt_path, "r").read()
        self.apply_rule_prompt = open(apply_rule_prompt_path, "r").read()
    
    def generate_all(self):
        for dialogue in tqdm(self.dialogues):
            print(f"Generating dialogue with ID {dialogue.id}")
            self.generate_single_dialogue(dialogue)
            print(f"Dialogue with ID {dialogue.id} generated.")
    
    def generate_single_dialogue(self, dialogue: Dialogue):
        raw_turns = dialogue.turns
        self.dfs_generation(dialogue.id, [], raw_turns, 0)
    
    def dfs_generation(self,
                       dialogue_id: str,
                       dpo_turns: list[DPOTurn],
                       raw_turns: list[Turn],
                       current: int) -> None:
        if current == len(raw_turns):
            print(f"Reached the end of the dialogue with ID {dialogue_id}")
            return
        
        # We need to know which rules should be applied or not
        upcoming_turn = raw_turns[current]
        applicable_rules = []
        for rule_idx, rule in self.rules:
            if self._does_rule_get_applied(rule_idx, dpo_turns, upcoming_turn):
                applicable_rules.append(rule_idx)

        # Now for each rule we will generate the dpo turn
        local_dpo_turns = []
        for rule_idx in applicable_rules:
            possible_doc_id = DPODialogue.get_id(dialogue_id, [turn.rule_used for turn in dpo_turns]+[rule_idx])
            if possible_doc_id in self.already_processed:
                continue
            # First the adapted student question and tutor response
            adapted_student, adapted_tutor = self._get_good_answer_and_question(
                dpo_turns[-1] if dpo_turns else None,
                rule_idx,
                upcoming_turn,
            )
            # Now the negative answer
            negative_answer = self._get_bad_answer(rule_idx, adapted_tutor)

            # And now let's generate the dpo turn
            dpo_turn = DPOTurn(
                student_question=adapted_student,
                positive_answer=adapted_tutor,
                negative_answer=negative_answer,
                rule_used=rule_idx
            )
            local_dpo_turns.append(dpo_turn)
        
        # Let's save everything generated so far
        for turn in local_dpo_turns:
            final_list = dpo_turns + [turn]
            new_dialogue_id = DPODialogue.get_id(dialogue_id, [turn.rule_used for turn in final_list])
            dialogue = DPODialogue(
                id=new_dialogue_id,
                last_turn=turn,
                output_jsonl=self.output_jsonl
            )
            dialogue.save()

        # Out of all the dpo turns we will select k at random to continue the generation
        selected_dpo_turns = random.sample(local_dpo_turns, min(self.K, len(local_dpo_turns)))
        # These turns will be used to continue the generation
        for turn in selected_dpo_turns:
            self.dfs_generation(dialogue_id, dpo_turns + [turn], raw_turns, current+1)

    def _generate_prompt_apply_rule(self,
                                    rule_index: int,
                                    dialogue_so_far: list[DPOTurn],
                                    upcoming_turn: Turn) -> str:
        prompt = self.apply_rule_prompt.replace("<PEDAGOGICAL RULE>", self.rules[rule_index])
        con_so_far = ""
        for turn in dialogue_so_far:
            con_so_far += f"Student: {turn.student_question}\n"
            con_so_far += f"Tutor: {turn.positive_answer}\n"
        # If the conversation is empty we will replace the placeholder with an empty string
        if con_so_far == "":
            con_so_far = "// the conversation has just started, no conversation so far\n"
        prompt = prompt.replace("<CONVERSATION SO FAR>", con_so_far)

        prompt = prompt.replace("<STUDENT QUESTION>", upcoming_turn.user)
        prompt = prompt.replace("<TUTOR ANSWER>", upcoming_turn.assistant)

        return prompt
    
    def _generate_prompt_good_answer_and_question(self,
                                                  last_dpo_turn: DPOTurn,
                                                  rule_to_apply: int,
                                                  upcoming_turn: Turn) -> str:
        prompt = self.good_answer_prompt.replace("<PEDAGOGICAL RULE>", self.rules[rule_to_apply])
        if last_dpo_turn is None:
            last_tut_res = "// the conversation has just started, no tutor’s response\n"
        else:
            last_tut_res = last_dpo_turn.positive_answer
        prompt = prompt.replace("<LAST TUTOR RESPONSE>", last_tut_res)
        prompt = prompt.replace("<STUDENT QUESTION>", upcoming_turn.user)
        prompt = prompt.replace("<TUTOR ANSWER>", upcoming_turn.assistant)
        return prompt
    
    def _generate_prompt_bad_answer(self,
                                    rule_to_negate: int,
                                    good_answer: str) -> str:
        prompt = self.bad_answer_prompt.replace("<PEDAGOGICAL RULE>", self.rules[rule_to_negate])
        prompt = prompt.replace("<GOOD TUTOR RESPONSE>", good_answer)
        return prompt
    
    def _get_bad_answer(self,
                        rule_to_negate: int,
                        good_answer: str) -> str:
        prompt = self._generate_prompt_bad_answer(rule_to_negate, good_answer)
        response = self._query_openai(prompt, BadAnswerSchema)
        return response["not_following_tutor"]

    
    def _get_good_answer_and_question(self,
                                      last_dpo_turn: DPOTurn,
                                      rule_to_apply: int,
                                      upcoming_turn) -> str:
        prompt = self._generate_prompt_good_answer_and_question(last_dpo_turn, rule_to_apply, upcoming_turn)
        response = self._query_openai(prompt, GoodAnswerSchema)
        return response["adapted_response"], response["tutor_response"]
    
    def _get_rule_scoring(self,
                               rule_to_apply: int,
                               dialogue_so_far: list[DPOTurn],
                               upcoming_turn) -> int:
        # If the rule has been applied in the past 3 turns, return False
        if rule_to_apply in [turn.rule_used for turn in dialogue_so_far[-3:]]:
            return False
        
        # Else we will ask OpenAI if the rule should be applied
        prompt = self._generate_prompt_apply_rule(rule_to_apply, dialogue_so_far, upcoming_turn)
        response = self._query_openai(prompt, UseRuleSchema)
        return response["rule_fit_score"]


    
    def _query_openai(self, prompt: str, schema: BaseModel) -> str:
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format=schema
        )
        # Parsing back to python object
        completion = completion.to_dict()
        response = completion["choices"][0]["message"]["parsed"]
        return response