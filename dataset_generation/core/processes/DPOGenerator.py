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

class DPOGenerator:
    K = 3 # The number of leafs to generate for each level of the dfs tree

    def __init__(self,
                 jsonl_file: str,
                 output_jsonl: str,
                 rules_txt_path: str,
                 good_answer_prompt_path: str,
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
        self.apply_rule_prompt = open(apply_rule_prompt_path, "r").read()
    
    def generate_all(self) -> None:
        for dialogue in tqdm(self.dialogues):
            try:
                self.generate_single_dialogue(dialogue)
            except Exception as e:
                print(f"Error while processing dialogue {dialogue.id}: {e}")
    
    def generate_single_dialogue(self, dialogue: Dialogue) -> None:
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
        
        upcoming_turn = raw_turns[current]
        rules_scores = []
        for rule_idx, rule in self.rules:
            score = self._get_rule_scoring(rule_idx, dpo_turns, upcoming_turn)
            # We do not want to apply rules that have a score of 3 or less
            if score not in [4, 5]:
                continue
            rules_scores.append((score, rule_idx))

        # If there are no rules to apply we will just skip this turn
        if not rules_scores:
            print(f"No applicable rules for dialogue {dialogue_id} at turn {current}. Skipping.")
            return

        # Isolating the rules that have the highest score [TODO] you can make it way more efficient
        rules_scores.sort(reverse=True)
        max_score = rules_scores[0][0]
        max_scoring_rules = [rule_idx for score, rule_idx in rules_scores if score == max_score]

        # Now we will randomly select K of the rules to apply
        applicable_rules = random.sample(max_scoring_rules, min(self.K, len(max_scoring_rules)))
        
        # Now for each rule we will generate the dpo turn
        local_dpo_turns = []
        for rule_idx in applicable_rules:
            possible_doc_id = DPODialogue.get_id(dialogue_id, [turn.rule_used for turn in dpo_turns]+[rule_idx])
            if possible_doc_id in self.already_processed:
                print(f"Skipping rule {rule_idx} for dialogue {dialogue_id} as it has already been processed.")
                continue
            # First the adapted student question and tutor response
            adapted_student, adapted_tutor = self._get_good_answer_and_question(
                dpo_turns[-1] if dpo_turns else None,
                rule_idx,
                upcoming_turn,
            )

            # And now let's generate the dpo turn
            dpo_turn = DPOTurn(
                student_question=adapted_student,
                positive_answer=adapted_tutor,
                negative_answer=upcoming_turn.assistant, # using as negative answer the original tutor response without any rule
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

        # Out of all the dpo turns generated so far, we will select only one random one to continue
        to_continue = random.choice(local_dpo_turns)
        self.dfs_generation(dialogue_id, dpo_turns + [to_continue], raw_turns, current+1)

    def _generate_prompt_apply_rule(self,
                                    rule_index: int,
                                    dialogue_so_far: list[DPOTurn],
                                    upcoming_turn: Turn) -> str:
        prompt = self.apply_rule_prompt.replace("<PEDAGOGICAL RULE>", self.rules[rule_index])
        con_so_far = ""
        for turn in dialogue_so_far[-2:]: # We only need the last two interactions to make a call
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
        # If the rule has been applied in the past 3 turns, return False [TODO] ask confirmation to proceed
        if rule_to_apply in [turn.rule_used for turn in dialogue_so_far[-3:]]:
            return 0
        
        # Else we will ask OpenAI if the rule should be applied
        prompt = self._generate_prompt_apply_rule(rule_to_apply, dialogue_so_far, upcoming_turn)
        response = self._query_openai(prompt, UseRuleSchema)
        score = response["rule_fit_score"]

        # Manually clipping the score between 1 and 5
        if score < 1:
            score = 1
        elif score > 5:
            score = 5

        return score

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