from core.processes import ChunkExtractor
from core.processes import DialogueGenerator
from core.processes import DPOGenerator
from core.loaders import DocumentLoader, DialogueLoader, DPODialogueLoader
from argparse import ArgumentParser

def start_generation(raw_pdfs: str,
                     dialogue_prompt: str,
                     rules_list: str,
                     good_answer_and_question_prompt: str,
                     choose_rule_prompt: str,
                     extracted_texts_json: str,
                     dialogues_json: str,
                     dpo_dialogues: str,
                     max_generations: int):

    # First you will need to extract the text from the pdfs, usually this is done in bulk once
    # This is done using the TextExtractor class
    extractor = ChunkExtractor(raw_pdfs, extracted_texts_json)
    extractor.extract_texts()

    dialogue_gen = DialogueGenerator(extracted_texts_json, dialogues_json, dialogue_prompt)
    dialogue_gen.generate_all(max_generations=max_generations)

    dpo_gen = DPOGenerator(
        dialogues_json,
        dpo_dialogues,
        rules_list,
        good_answer_and_question_prompt,
        choose_rule_prompt
    )
    dpo_gen.generate_all()

def print_statistics(extracted_texts_json: str, dialogues_json: str, dpo_dialogues: str):
    doc_loader = DocumentLoader(extracted_texts_json)
    dialogue_loader = DialogueLoader(dialogues_json)
    dpo_dialogue_loader = DPODialogueLoader(dpo_dialogues)

    print(f"Number of documents: {doc_loader}")
    print(f"Number of dialogues: {dialogue_loader}")
    print(f"Number of DPO dialogues: {dpo_dialogue_loader}")

    # Counting all words in all documents
    total_words = 0
    for i in range(len(doc_loader)):
        for chunk in doc_loader[i].chunks:
            total_words += len(chunk.text.split())
    print(f"Total number of words in all documents: {total_words}")

    # Counting all words in all dialogues
    total_words = 0
    for i in range(len(dialogue_loader)):
        turns = dialogue_loader[i].turns
        for turn in turns:
            total_words += len(turn.assistant.split(" "))
    print(f"Total number of words in all dialogues (just assistant): {total_words}")

    # Counting all words in all DPO dialogues
    total_words = 0
    for dpo_id in dpo_dialogue_loader.get_unique_dpo_ids():
        turns = dpo_dialogue_loader.get_dpo_turns_by_dialogue_id(dpo_id)
        for turn in turns:
            total_words += len(turn.positive_answer.split(" "))
    print(f"Total number of words in all DPO dialogues (positive answers): {total_words}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_pdfs", type=str, required=True)
    parser.add_argument("--dialogue_prompt", type=str, required=True)
    parser.add_argument("--rules_list", type=str, required=True)
    parser.add_argument("--good_answer_and_question_prompt", type=str, required=True)
    parser.add_argument("--choose_rule_prompt", type=str, required=True)
    parser.add_argument("--extracted_texts_json", type=str, required=True)
    parser.add_argument("--dialogues_json", type=str, required=True)
    parser.add_argument("--dpo_dialogues_json", type=str, required=True)
    parser.add_argument("--max_generations", type=int, required=True)
    args = parser.parse_args()

    start_generation(
        args.raw_pdfs,
        args.dialogue_prompt,
        args.rules_list,
        args.good_answer_and_question_prompt,
        args.choose_rule_prompt,
        args.extracted_texts_json,
        args.dialogues_json,
        args.dpo_dialogues_json,
        args.max_generations
    )
    print_statistics(args.extracted_texts_json, args.dialogues_json, args.dpo_dialogues_json)