import argparse
import os
import json
import time
from Prompter.prompter import Prompter

# GPT-4.5-Preview, 
ALL_LLMS = ['Command A (03-2025)', 'Gemini-2.0-Flash-Thinking-Exp-01-21', 'Gemini-1.5-Pro-002', 
            'Llama-3.3-Nemotron-Super-49B-v1', 'DeepSeek-R1', 'ChatGPT-4o-latest (2025-01-29)', 
            'DeepSeek-V3', 'Qwen-Plus-0125', 'o3-mini-high', 'o1-2024-12-17', 'Qwen2.5-Max', 
            'Mistral-Large-2407', 'Gemma-3-27B-it', 'Deepseek-v2.5-1210', 
            'Claude 3.7 Sonnet (thinking-32k)', 'Meta-Llama-3.1-405B-Instruct-bf16', 
            'QwQ-32B', 'GPT-4.5-Preview', 'Athene-v2-Chat-72B', 'o1-mini']

ALL_TEMPERATURES = [0.0, 0.3, 0.7, 1.0]

MODEL_CONVERTER = {
    "GPT-4.5-Preview": "gpt-4.5-preview-2025-02-27",
    "ChatGPT-4o-latest (2025-01-29)": "chatgpt-4o-latest",
    "o1-2024-12-17": "o1-2024-12-17",
    "o3-mini-high": "o3-mini",
    "o1-mini": "o1-mini",
    "DeepSeek-R1": "deepseek-reasoner",
    "DeepSeek-V3": "deepseek-chat",
    "Gemini-1.5-Pro-002" : "gemini-1.5-pro-002",
    "Gemini-2.0-Flash-Thinking-Exp-01-21": "gemini-2.0-flash-thinking-exp-01-21",
    "Claude 3.7 Sonnet (thinking-32k)": "claude-3-7-sonnet-20250219",
    "Gemma-3-27B-it": "google/gemma-3-27b-it",
    "QwQ-32B": "Qwen/QwQ-32B",
}

def parse_json(input_path, output_path, selected_llms=ALL_LLMS, selected_temperatures=ALL_TEMPERATURES, delay=None):
    with open(input_path, "r") as file:
        data = json.load(file)

    system_prompt = data["system_prompt"]

    # Iterate through each LLM in the JSON data
    for llms in data["prompts"]:
        llm = llms["llm"]
        if llm not in selected_llms:
            print(f"Skipping LLM: {llm}")
            continue
        for temperatures in llms["prompts"]:
            temperature = temperatures["temperature_level"]
            if temperature not in selected_temperatures:
                print(f"Skipping temperature level: {temperature}")
                continue
            for cq_rep in temperatures["comprehension_questions"]:
                for cq in cq_rep["prompts"]:
                    if 'answer' not in cq:
                        cq["answer"] = send_prompt(
                            llm=llm,
                            temperature=temperature,
                            format_answer=cq["format_answer"],
                            system_prompt=system_prompt,
                            prompt=cq["prompt"],
                        )
                        if delay:
                            time.sleep(int(delay))


            for task_rep in temperatures["tasks"]:
                for task in task_rep["prompts"]:
                    if 'answer' not in cq:
                        task["answer"] = send_prompt(
                            llm=llm,
                            temperature=temperature,
                            format_answer=task["format_answer"],
                            system_prompt=system_prompt,
                            prompt=task["prompt"],
                            role=task["role"],
                            participant=task["participant"]
                        )
                        if delay:
                            time.sleep(int(delay))


        ## write the modified data back to the JSON file
        if output_path == None:
            output_path = input_path
        with open(output_path, "w") as file:
            json.dump(data, file, indent=4)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to the input JSON file")
    # parser.add_argument("-o", "--output_path", type=str, help="Path to the output JSON file")
    # parser.add_argument("-p", "--provider_config", type=str, help="Path to the provider config file")
    # parser.add_argument("--llm", type=str, choices=ALL_LLMS, help="LLM to use")
    # parser.add_argument("--temp", type=float, choices=ALL_TEMPERATURES, help="Temperature to use")
    # parser.add_argument("--delay", type=str, help="How long to wait in seconds(s) before sending the next prompt")
    # args = parser.parse_args()

    # if os.path.isfile(args.output_path) == False:
    #     os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    #     print(f"Output path {args.output_path} does not exist. Creating it.")

    # if args.llm:
    #     selected_llms = [args.llm]
    # else:
    #     selected_llms = ALL_LLMS

    # for llm in selected_llms:
    #     if llm not in MODEL_CONVERTER:
    #         raise ValueError(f"Unsupported LLM: {llm}")
    
    # if args.temp:
    #     selected_temperatures = [args.temp]

    # parse_json(
    #     input_path=args.input_path,
    #     output_path=args.output_path,
    #     selected_llms=selected_llms,
    #     selected_temperatures=selected_temperatures,
    #     delay=args.delay
    # )

    prompter = Prompter(dotenv_path=".env", provider_metadata_path='./llm_provider_config.json')
    llm = MODEL_CONVERTER["Gemini-2.0-Flash-Thinking-Exp-01-21"]
    llm = "Qwen/QwQ-32B"
    temperature = 0.7
    resp = prompter.send_prompt(llm=llm, temperature=temperature, 
                                system_prompt="You are a helpful assistant.", 
                                prompt="What is the capital of France?")
    print(resp)
