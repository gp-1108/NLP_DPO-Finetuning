"""
Author: gp-1108
Description: This small script here is meant to convert the DeLorenzi dataset from its original format
to a more generalized chat format that is used by Huggingface. Each interaction is modeled as a list of
dictionaries, where each dictionary represents a turn in the conversation. The dictionary contains the
following keys:
    - role: the role of the speaker (system, assistant, user)
    - content: the content of the message

A sample conversation in this format is as follows:
[
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {
        "role": "user",
        "content": "How many helicopters can a human eat in one sitting?"
    },
]
"""
DELORENZI_CONV_SEPARATOR="<---------->"
DELORENZI_TURN_SEPARATOR="<MARK>"

import json
import os
import argparse

def load_delorenzi_txt(path: str) -> list[list[str]]:
    """
    This function loads the txt file and returns all the interactions in a list of lists.
    Each list represents a conversation, each element in the list represents a turn in the conversation.

    Args:
        - path: the path to the txt file

    Returns:
        - a list of lists, where each list represents a conversation
    """

    text = open(path, "r").read()
    conversations = text.split(DELORENZI_CONV_SEPARATOR)
    conversations = conversations[:-1] # The last element is an empty string
    conversations = [conv.split(DELORENZI_TURN_SEPARATOR) for conv in conversations]
    return conversations

def convert_to_chat_format(conversations: list[list[str]]) -> list[list[dict]]:
    """
    This function converts the conversations to the chat format.

    Args:
        - conversations: a list of lists, where each list represents a conversation

    Returns:
        - a list of lists, where each list represents a conversation in chat format
    """

    chats = []
    for conversation in conversations:
        chat = []
        for i, turn in enumerate(conversation):
            turn = turn.strip()
            turn = turn.replace("</s>", "")
            if i % 2 == 0:
                role = "user"
            else:
                role = "assistant"
            chat.append({
                "role": role,
                "content": turn.strip()
            })
        chats.append(chat)
    
    return chats

def save_to_json(chats: list[list[dict]], path: str):
    """
    This function saves the chats to a json file.

    Args:
        - chats: a list of lists, where each list represents a conversation in chat format
    
    Returns:
        - None
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, "w") as f:
        json.dump(chats, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the DeLorenzi txt file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output json file")
    args = parser.parse_args()

    conversations = load_delorenzi_txt(args.input)
    chats = convert_to_chat_format(conversations)
    save_to_json(chats, args.output)

if __name__ == "__main__":
    main()