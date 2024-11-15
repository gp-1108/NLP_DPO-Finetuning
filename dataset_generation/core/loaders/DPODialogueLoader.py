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
