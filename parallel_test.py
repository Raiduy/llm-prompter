from multiprocessing import Pool, current_process, cpu_count
from Prompter.prompter import Prompter

import argparse
import json
import os
import time

# CONSTANTS
CHECKPOINT_FOLDER = './pll_checkpoints'
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
    "Athene-v2-Chat-72B": "Athene-v2-Chat-72B",
    "Deepseek-v2.5-1210": "deepseek-v2.5:latest",
    "Meta-Llama-3.1-405B-Instruct-bf16": "meta-llama/Meta-Llama-3.1-405B-Instruct",
}


def write_json(data, output_path):
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)

def remove_smaller_checkpoints(llm):
    print(f'Removing small checkpoints for {llm}')
    for file in os.listdir(CHECKPOINT_FOLDER):
        if f'{llm}_temp_' in file:
            os.remove(os.path.join(CHECKPOINT_FOLDER, file))


def process_prompts(prompter, system_prompt, llm, temperature, prompt_repetition, prompt_category, delay):
    needs_checkpoint = False
    for i, prompt in enumerate(prompt_repetition["prompts"]):
        total_prompts = len(prompt_repetition["prompts"])
        if 'answer' not in prompt:
            answer = 'Test'
            # answer = prompter.send_prompt(
            #     llm=MODEL_CONVERTER[llm],
            #     temperature=temperature,
            #     system_prompt=system_prompt,
            #     prompt=prompt["prompt"],
            # )
            if answer != None:
                prompt["answer"] = answer
                print(f'Reply received for {llm}, {temperature}, {prompt["id"]}, \
                      rep {prompt_repetition["repetition_id"]} for \
                      {prompt_category} prompt {i+1}/{total_prompts}')
                needs_checkpoint = True
                write_json(prompt_repetition, \
                           f'{CHECKPOINT_FOLDER}/{llm}_temp_{temperature}_rep_{prompt_repetition["repetition_id"]}_{prompt_category}.json')
                if delay:
                    print(f"Waiting for {delay} seconds before sending the next prompt...")
                    time.sleep(int(delay))
        else:
            print(f'Answer already exists for {llm}, {temperature}, {prompt["id"]}, \
                  rep {prompt_repetition["repetition_id"]} for \
                  {prompt_category} prompt {i+1}/{total_prompts}')

    return prompt_repetition, needs_checkpoint


def process_llm(provider_config, llm_data, system_prompt, delay=None):
    prompter = Prompter(dotenv_path='.env', provider_metadata_path=provider_config)
    # Iterate through each LLM in the JSON data
    llm = llm_data["llm"]
            
    for temperatures in llm_data["prompts"]:
        temperature = temperatures["temperature_level"]

        if "using_default_temperature" in llm_data and temperature != 0.0:
            print(f"Model does not support temperatures")
            continue

        for cq_rep in temperatures["comprehension_questions"]:
            cq_rep, cq_needs_checkpoint = process_prompts(prompter, system_prompt, llm, temperature, cq_rep, "cq", delay)

        for task_rep in temperatures["tasks"]:
            task_rep, task_needs_checkpoint = process_prompts(prompter, system_prompt, llm, temperature, task_rep, "task", delay)

    if cq_needs_checkpoint or task_needs_checkpoint:
        write_json(llm_data, f'{CHECKPOINT_FOLDER}/{llm}.json')
        remove_smaller_checkpoints(llm)
    return llm_data
       

def process_content(parameter_list):
    print(f'LLM {parameter_list[0]["llm"]} is processed by process {current_process().name}')
    data_contents = parameter_list[0]
    sys_prompt = parameter_list[1]
    provider_config = parameter_list[2]
    delay = parameter_list[3]
    return process_llm(provider_config, data_contents, sys_prompt, delay=delay)


def process_arguments(args):
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

    if os.path.exists(args.input_path) == False:
        raise ValueError(f'FilePath {args.input_path} does not exist')
    
    if os.path.exists(args.provider_config) == False:
        raise ValueError(f'FilePath {args.provider_config} does not exist')
    
    # Set default values
    selected_llms = list(MODEL_CONVERTER.keys())

    # Validate user-selected values
    if args.llm:
        selected_llms = [args.llm]

    for llm in selected_llms:
        if llm not in list(MODEL_CONVERTER.keys()):
            print(f"Unsupported LLM: {llm}, skipping...")
            selected_llms.remove(llm)

    return selected_llms
    

def generate_parallelizable_datastructure(input_path, provider_config, delay, selected_llms):
    with open(input_path, 'r') as f:
        data = json.load(f)

    sys_prompt = data["system_prompt"]

    to_parallelize = []
    llm_list = []
    for llms in data["prompts"]:
        if llms["llm"] in selected_llms:
            to_parallelize.append([llms, sys_prompt, provider_config, delay])
            llm_list.append(llms["llm"])
        else:
            print(f'Skipping LLM {llms["llm"]}')

    print(f'Running on LLMs: {llm_list}')
    parallelized_data = tuple(to_parallelize)
    return parallelized_data, sys_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output JSON file")
    parser.add_argument("-p", "--provider_config", type=str, help="Path to the provider config file")
    parser.add_argument("--llm", type=str, choices=list(MODEL_CONVERTER.keys()), help="LLM to use")
    parser.add_argument("--delay", type=str, help="How long to wait in seconds(s) before sending the next prompt")
    args = parser.parse_args()

    llms = process_arguments(args)
    parallelized_data, sys_prompt = generate_parallelizable_datastructure(args.input_path, args.provider_config, args.delay, llms)

    num_cpus = cpu_count() - 4  # Leaving some threads open :)
    if num_cpus > len(llms):
        num_cpus = len(llms)

    with Pool(processes=num_cpus) as pool:
        results = pool.map(process_content, parallelized_data)
    
    output_dict = {
        "system_prompt": sys_prompt,
        "prompts": results
    }

    if not args.output_path:
        args.output_path = args.input_path
    write_json(output_dict, args.output_path)
