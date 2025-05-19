import argparse
import os
import json
import time
from Prompter.prompter import Prompter

# GPT-4.5-Preview, 
ALL_LLMS = ['Command A (03-2025)', 'Gemini-2.0-Flash-Thinking-Exp-01-21', 'Gemini-1.5-Pro-002', 
            'Llama-3.3-Nemotron-Super-49B-v1', 'DeepSeek-R1', 'ChatGPT-4o-latest (2025-01-29)', 
            'DeepSeek-V3', 'Qwen-Plus-0125', 'o3-mini', 'o1-2024-12-17', 'Qwen2.5-Max', 
            'Mistral-Large-2407', 'Gemma-3-27B-it', 'Deepseek-v2.5-1210', 
            'Claude 3.7 Sonnet (thinking-32k)', 'Meta-Llama-3.1-405B-Instruct-bf16', 
            'QwQ-32B', 'GPT-4.5-Preview', 'Athene-v2-Chat-72B', 'o1-mini']

ALL_TEMPERATURES = [0.0, 0.3, 0.7, 1.0]

MODEL_CONVERTER = {
    "GPT-4.5-Preview": "gpt-4.5-preview-2025-02-27",
    "ChatGPT-4o-latest (2025-01-29)": "chatgpt-4o-latest",
    "o1-2024-12-17": "o1-2024-12-17",
    "o3-mini": "o3-mini",
    "o1-mini": "o1-mini",
    "DeepSeek-R1": "deepseek-reasoner",
    "DeepSeek-V3": "deepseek-chat",
    "Gemini-1.5-Pro-002" : "gemini-1.5-pro-002",
    "Gemini-2.0-Flash-Thinking-Exp-01-21": "gemini-2.0-flash-thinking-exp-01-21",
    "Claude 3.7 Sonnet (thinking-32k)": "claude-3-7-sonnet-20250219",
    "Gemma-3-27B-it": "gemma-3-27b-it",
    "QwQ-32B": "QwQ-32B",
    "Llama-3.3-Nemotron-Super-49B-v1": "nvidia/llama-3.3-nemotron-super-49b-v1",
    "Qwen2.5-Max": "qwen-max-0125",
    "Qwen-Plus-0125": "qwen-plus-0125",
    "Mistral-Large-2407": "mistral-large-2407",
    "Command A (03-2025)": "command-a-03-2025",
    # "Athene-v2-Chat-72B": "Athene-v2-Chat-72B",
    # "Deepseek-v2.5-1210": "deepseek-v2.5-1210",
    "Meta-Llama-3.1-405B-Instruct-bf16": "meta-llama/Meta-Llama-3.1-405B-Instruct",
}


def write_json(data, output_path):
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)


def parse_json(prompter, input_path, output_path, selected_llms=ALL_LLMS, selected_temperatures=ALL_TEMPERATURES, delay=None):
    with open(input_path, "r") as file:
        data = json.load(file)

    if output_path == None:
            output_path = input_path

    system_prompt = data["system_prompt"]

    # Iterate through each LLM in the JSON data
    for llms in data["prompts"]:
        llm = llms["llm"]
        if llm not in selected_llms or llm not in MODEL_CONVERTER:
            print(f"Skipping LLM: {llm}")
            continue
        
        TEMP_FLAG = "custom"
        if "using_default_temperature" in llms:
            TEMP_FLAG = "default"

        for temperatures in llms["prompts"]:
            temperature = temperatures["temperature_level"]

            if TEMP_FLAG == "default" and temperature != 0.0:
                print(f"Model does not support temperatures")
                continue

            if temperature not in selected_temperatures:
                print(f"Skipping temperature level: {temperature}")
                continue
            

            for cq_rep in temperatures["comprehension_questions"]:
                for i, cq in enumerate(cq_rep["prompts"]):
                    total_prompts = len(cq_rep['prompts'])
                    if 'answer' in cq:
                        print(f"Answer already exists for {llm}, {temperature}, {cq['id']}, rep {cq_rep["repetition_id"]} for comprehension prompt {i+1}/{total_prompts}")
                        continue
                    else:
                        answer = prompter.send_prompt(
                            llm=MODEL_CONVERTER[llm],
                            temperature=temperature,
                            system_prompt=system_prompt,
                            prompt=cq["prompt"],
                        )
                        if answer != None:
                            cq["answer"] = answer
                            print(f"Answer given for {llm}, {temperature}, {cq['id']}, rep {cq_rep["repetition_id"]} for comprehension prompt {i+1}/{total_prompts}")
                            write_json(cq_rep, f'./checkpoints/{llm}_temp_{temperature}_rep_{cq_rep["repetition_id"]}_cq.json')
                            write_json(data, output_path)  # Save the modified data to the output path
                            if delay:
                                print(f"Waiting for {delay} seconds before sending the next prompt...")
                                time.sleep(int(delay))

            for task_rep in temperatures["tasks"]:
                for i, task in enumerate(task_rep["prompts"]):
                    total_prompts = len(task_rep['prompts'])
                    if 'answer' in task:
                        print(f"Answer already exists for {llm}, {temperature}, {task['id']}, rep {task_rep['repetition_id']} for task prompt {i+1}/{total_prompts}")
                        continue
                    else:
                        answer = prompter.send_prompt(
                            llm=MODEL_CONVERTER[llm],
                            temperature=temperature,
                            system_prompt=system_prompt,
                            prompt=task["prompt"],
                        )
                        if answer != "Model not found in any provider.":
                            task["answer"] = answer
                            print(f"Answer given for {llm}, {temperature}, {task['id']}, rep {task_rep['repetition_id']} for task prompt {i+1}/{total_prompts}")
                            write_json(task_rep, f'./checkpoints/{llm}_temp_{temperature}_rep_{task_rep["repetition_id"]}_task.json')
                            write_json(data, output_path)  # Save the modified data to the output path
                            if delay:
                                print(f"Waiting for {delay} seconds before sending the next prompt...")
                                time.sleep(int(delay))
    
    # write_json(data, output_path)  # Save the modified data to the output path

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output JSON file")
    parser.add_argument("-p", "--provider_config", type=str, help="Path to the provider config file")
    parser.add_argument("--llm", type=str, choices=ALL_LLMS, help="LLM to use")
    parser.add_argument("--temp", type=float, choices=ALL_TEMPERATURES, help="Temperature to use")
    parser.add_argument("--delay", type=str, help="How long to wait in seconds(s) before sending the next prompt")
    args = parser.parse_args()

    if args.output_path and os.path.isfile(args.output_path) == False:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        print(f"Output path {args.output_path} does not exist. Creating it.")

    selected_llms = ALL_LLMS
    selected_temperatures = ALL_TEMPERATURES

    if args.llm:
        selected_llms = [args.llm]

    if args.temp:
        selected_temperatures = [args.temp]

    print(f"Selected LLMs: {selected_llms}")
    print(f"Selected Temperatures: {selected_temperatures}")
        
    for llm in selected_llms:
        if llm not in MODEL_CONVERTER:
            print(f"Unsupported LLM: {llm}, skipping...")
            selected_llms.remove(llm)
            continue

    prompter = Prompter(dotenv_path=".env", provider_metadata_path='./llm_provider_config.json')
    parse_json(prompter, args.input_path, args.output_path, selected_llms=selected_llms, 
               selected_temperatures=selected_temperatures, delay=args.delay)
    
    # llm = MODEL_CONVERTER["Command A (03-2025)"]
    # # llm = "Qwen/QwQ-32B"
    # temperature = 0.0
    # resp = prompter.send_prompt(llm=llm, temperature=temperature, 
    #                             system_prompt="You are a helpful assistant.", 
    #                             prompt="What was my last question?")
    # print(resp)
