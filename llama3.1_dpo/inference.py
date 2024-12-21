import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftConfig, PeftModel
from transformers import TextIteratorStreamer
from threading import Thread
import time
import os

# Function to format the conversation prompt
def format_prompt(conversation):
    """
    Formats a conversation into the model's prompt structure.

    Args:
        conversation (list of dict): The chat history.

    Returns:
        str: Formatted prompt.
    """
    if len(conversation) == 0:
        return "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n"

    prompt = "<|begin_of_text|>"
    for turn in conversation:
        prompt += f"<|start_header_id|>{turn['role']}<|end_header_id|>\n{turn['content']}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

# Function to generate a response from the model
def generate_ans_streamed(prompt, llm_model, tokenizer, device='cuda', max_new_tokens=2048, temperature=0.1, verbose=False):
    """
    Generates a response from the LLM model using a streaming approach.

    Args:
        prompt (str): The formatted prompt.
        llm_model (transformers.PreTrainedModel): The language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        device (str): Device to run the model on ('cuda' or 'cpu').
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature for response generation.
        verbose (bool): If True, prints useful statistics.

    Returns:
        str: Generated response.
    """
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id

    tokenized = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    start_time = time.time()
    thread = Thread(target=llm_model.generate, kwargs={
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature
    })
    thread.start()

    response = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        response += new_text

    thread.join()
    end_time = time.time()

    if verbose:
        response_tokens = len(tokenizer(response).input_ids)
        print(f"\n[Statistics] Response Time: {end_time - start_time:.2f} seconds | Response Length: {response_tokens} tokens")

    print()  # Add a newline after the response
    return response

# Main function for the interactive chatbot
def main(args):
    peft_model_id = args.model_path
    config = PeftConfig.from_pretrained(peft_model_id)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.eos_token = "<|eot_id|>"
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, peft_model_id)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"

    conversation = []

    print("\nWelcome to the CLI Chatbot! Type 'clear' to reset the chat history or 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            conversation.clear()
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal screen
            print("Chat history cleared!\n")
            continue

        conversation.append({"role": "user", "content": user_input})
        prompt_text = format_prompt(conversation)

        print("Assistant: ", end="")
        response = generate_ans_streamed(
            prompt_text, 
            model, 
            tokenizer, 
            device, 
            max_new_tokens=args.max_new_tokens, 
            temperature=args.temperature, 
            verbose=args.verbose
        )
        conversation.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive chatbot with CLI and streaming responses.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the folder containing the adapter_config.json and .bin files.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode to display useful statistics.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for response generation (default: 0.1).")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of new tokens to generate (default: 2048).")
    args = parser.parse_args()

    main(args)
