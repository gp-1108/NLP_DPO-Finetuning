class PedagogicalRules:
    """
    A class to handle pedagogical rules loaded from a text file.
    This class manages a collection of pedagogical rules, where each rule has an arbitrary index.
    The rules are loaded from a text file where each line contains a rule index followed by the rule text.
    The class provides bidirectional mapping between rule indices and rule texts, and supports iteration
    over the rules.
    Attributes:
        idx2rules (dict): A dictionary mapping rule indices to rule texts
        rules2idx (dict): A dictionary mapping rule texts to rule indices
    Methods:
        __iter__: Allows iteration over the rules
        __next__: Returns the next rule during iteration
        __getitem__: Enables dictionary-like access to rules using either index or text
    Example:
        rules = PedagogicalRules("rules.txt")
        # Get rule text by index
        rule_text = rules[1]
        # Get rule index by text
        rule_index = rules["Some rule text"]
        # Iterate over rules
        for idx, text in rules:
            print(f"Rule {idx}: {text}")
    """ 
    def __init__(self, rules_txt_path: str):
        """
        Initialize the PedagogicalRules object.
        The rules txt file should contain one rule per line, with the rule index at the beginning of the line.
        """
        self.idx2rules, self.rules2idx = PedagogicalRules._load_rules(rules_txt_path)
        self._iterator_index = 0  # Initialize the iterator index
    
    @staticmethod
    def _load_rules(rules_txt_path: str) -> tuple[dict[str, str], dict[str, str]]:
        idx2rules = {}
        rules2idx = {}
        with open(rules_txt_path, "r") as f:
            for line in f:
                rule = line.strip()
                # The first word is the index of the rule
                rule_index = int(rule.split(" ")[0])
                rule_text = rule.replace(f"{rule_index} ", "")
                idx2rules[rule_index] = rule_text
                rules2idx[rule_text] = rule_index
        return idx2rules, rules2idx
    
    def __iter__(self):
        self._iterator_index = 0  # Reset iterator index
        self._rule_keys = list(self.idx2rules.keys())  # Create an ordered list of rule keys for iteration
        return self
    
    def __next__(self):
        if self._iterator_index < len(self._rule_keys):
            rule_index = self._rule_keys[self._iterator_index]
            rule_text = self.idx2rules[rule_index]
            self._iterator_index += 1
            return rule_index, rule_text  # Return a tuple (index, rule text)
        else:
            raise StopIteration  # End iteration
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.idx2rules[key]
        elif isinstance(key, str):
            return self.rules2idx[key]
        else:
            raise KeyError("Key must be an integer or a string")
    