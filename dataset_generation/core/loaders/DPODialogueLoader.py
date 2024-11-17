from ..components.DPODialogue import DPODialogue
from .BaseLoader import BaseLoader
import os

class DPODialogueLoader(BaseLoader):
    def __init__(self, jsonl_path):
        super().__init__(jsonl_path)
    
    def load_data(self):
        if not os.path.exists(self.jsonl_path):
            return []
        with open(self.jsonl_path, 'r') as file:
            data = [DPODialogue(json_str=line) for line in file if line.strip()]
        self.id2idx = {dialogue.id: idx for idx, dialogue in enumerate(data)}
        return data
    
    def load_index(self):
        index = {dialogue.id for dialogue in self.data}
        return index
    
    def get_unique_dpo_ids(self):
        # Basically we have a list of DPODialogue objects, their ids are structured as follows:
        # dc1_ch[0_1]_dpo[11_19_7_25]
        # dc1_ch[0_1]_dpo[11_19_4]
        # dc1_ch[0_1]_dpo[7]
        # dc1_ch[0_1]_dpo[11]
        # We would like to save only the longest id, so we can get the unique DPO ids
        uniques = set()
        for dialogue in self.data:
            id = dialogue.id
            if id in uniques:
                continue
            uniques.add(id)

            prev = DPODialogue.get_previous_dpo_id(id)
            while prev: # REALLY INNEFICIENT, maybe come up with a better way to do this [TODO]
                uniques.discard(prev)
                prev = DPODialogue.get_previous_dpo_id(prev)
        lst = list(uniques)
        # Now sort by length and then by the id itself
        lst.sort(key=lambda x: (len(x), x))
        return lst
    
    def get_dpo_turns_by_dialogue_id(self, dpo_dialogue_id):
        dpo_dialogue = self.get_dpo_dialogue_by_id(dpo_dialogue_id)
        turns = [dpo_dialogue.last_turn]
        prev = DPODialogue.get_previous_dpo_id(dpo_dialogue_id)
        while prev:
            dpo_dialogue = self.get_dpo_dialogue_by_id(prev)
            turns.append(dpo_dialogue.last_turn)
            prev = DPODialogue.get_previous_dpo_id(prev)
        return turns[::-1]
    
    def get_dpo_dialogue_by_id(self, dpo_dialogue_id):
        return self.data[self.id2idx[dpo_dialogue_id]]

    
    def get_dpo_dialogues_by_dialogue_id(self, dialogue_id):
        unique_dpo_ids = self.get_unique_dpo_ids()
        dpo_dialogues = [dpo_id for dpo_id in unique_dpo_ids if dpo_id.startswith(dialogue_id)]
        dpo_dialogues = [self.data[self.id2idx[dpo_id]] for dpo_id in dpo_dialogues]
        return dpo_dialogues
