from ..components import DPODialogue, DPOTurn
from .BaseLoader import BaseLoader
import os

class DPODialogueLoader(BaseLoader):
    def __init__(self, jsonl_path):
        super().__init__(jsonl_path)
    
    def load_data(self) -> list[DPODialogue]:
        """
        Loads DPODialogue data from a JSONL file.

        The method reads dialogue data from a JSONL file and creates DPODialogue objects for each line.
        Also builds an index mapping dialogue IDs to their position in the data list.

        Returns:
            list: List of DPODialogue objects created from the JSONL file data.
                  Returns empty list if file doesn't exist.

        Side Effects:
            Sets self.id2idx mapping dialogue IDs to indices in the returned list.
        """
        if not os.path.exists(self.jsonl_path):
            return []
        with open(self.jsonl_path, 'r') as file:
            data = [DPODialogue(json_str=line) for line in file if line.strip()]
        self.id2idx = {dialogue.id: idx for idx, dialogue in enumerate(data)}
        return data
    
    def load_index(self):
        """
        Returns a set containing all dialogue IDs from the loaded data.
        Used by the __contains__ method to check if a dialogue ID exists in the dataset.

        Returns:
            set: A set of dialogue IDs extracted from the data collection.
        """
        index = {dialogue.id for dialogue in self.data}
        return index
    
    def get_unique_dpo_ids(self) -> list[str]:
        """
        Extracts unique DPO IDs from dialogue data, keeping only the longest version of each ID.

        For example, given IDs like:
            - dc1_ch[0_1]_dpo[11_19_7_25]
            - dc1_ch[0_1]_dpo[11_19_4]
            - dc1_ch[0_1]_dpo[7]
            - dc1_ch[0_1]_dpo[11]

        It will only keep the longest version that encompasses shorter IDs
        (e.g. keeps dc1_ch[0_1]_dpo[11_19_7_25] but discards dc1_ch[0_1]_dpo[11]).

        Returns:
            list: A sorted list of unique DPO IDs, ordered first by length and then alphabetically.
        """
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
    
    def get_dpo_turns_by_dialogue_id(self, dpo_dialogue_id: str) -> list[DPOTurn]:
        """
        Get a list of DPOTurn objects for a given DPO dialogue ID, ordered chronologically.

        Given a DPO dialogue ID, retrieves all turns in the dialogue chain by traversing
        backwards through previous dialogues and collecting their last turns. The turns 
        are returned in chronological order (oldest to newest).

        Args:
            dpo_dialogue_id (str): The ID of the DPO dialogue to get turns for

        Returns:
            list[DPOTurn]: List of DPOTurn objects in chronological order
        """
        dpo_dialogue = self.get_dpo_dialogue_by_id(dpo_dialogue_id)
        turns = [dpo_dialogue.last_turn]
        prev = DPODialogue.get_previous_dpo_id(dpo_dialogue_id)
        while prev:
            dpo_dialogue = self.get_dpo_dialogue_by_id(prev)
            turns.append(dpo_dialogue.last_turn)
            prev = DPODialogue.get_previous_dpo_id(prev)
        return turns[::-1]
    
    def get_dpo_dialogue_by_id(self, dpo_dialogue_id: str) -> DPODialogue:
        """
        Retrieves a DPODialogue object by its unique identifier.

        Args:
            dpo_dialogue_id (str): The unique identifier of the DPODialogue to retrieve.

        Returns:
            DPODialogue: The DPODialogue object corresponding to the given ID.

        Raises:
            KeyError: If the dpo_dialogue_id is not found in the id2idx mapping.
        """
        return self.data[self.id2idx[dpo_dialogue_id]]

    
    def get_dpo_dialogues_by_dialogue_id(self, dialogue_id: str) -> list[DPODialogue]:
        """
        Retrieves all DPO dialogues that match a given dialogue ID prefix.

        Args:
            dialogue_id (str): The dialogue ID prefix to search for.

        Returns:
            list: A list of dialogue objects whose IDs start with the given dialogue_id.
                  Each dialogue object is retrieved from the data using the stored ID-to-index mapping.
        """
        unique_dpo_ids = self.get_unique_dpo_ids()
        dpo_dialogues = [dpo_id for dpo_id in unique_dpo_ids if dpo_id.startswith(dialogue_id)]
        dpo_dialogues = [self.data[self.id2idx[dpo_id]] for dpo_id in dpo_dialogues]
        return dpo_dialogues
