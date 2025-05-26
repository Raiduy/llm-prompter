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
    "Athene-v2-Chat-72B": "Athene-v2-Chat-72B",
    "Deepseek-v2.5-1210": "deepseek-v2.5:latest",
    "Meta-Llama-3.1-405B-Instruct-bf16": "meta-llama/Meta-Llama-3.1-405B-Instruct",
}


def write_json(data, output_path):
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)


def parse_json(prompter, input_path, output_path=None, selected_llms=ALL_LLMS, selected_temperatures=ALL_TEMPERATURES, delay=None):
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
                        print(f"Answer already exists for {llm}, {temperature}, {cq['id']}, rep {cq_rep['repetition_id']} for comprehension prompt {i+1}/{total_prompts}")
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
                            print(f"Answer given for {llm}, {temperature}, {cq['id']}, rep {cq_rep['repetition_id']} for comprehension prompt {i+1}/{total_prompts}")
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to the input JSON file")
    # parser.add_argument("-o", "--output_path", type=str, help="Path to the output JSON file")
    # parser.add_argument("-p", "--provider_config", type=str, help="Path to the provider config file")
    # parser.add_argument("--llm", type=str, choices=ALL_LLMS, help="LLM to use")
    # parser.add_argument("--temp", type=float, choices=ALL_TEMPERATURES, help="Temperature to use")
    # parser.add_argument("--delay", type=str, help="How long to wait in seconds(s) before sending the next prompt")
    # args = parser.parse_args()

    # if args.output_path and os.path.isfile(args.output_path) == False:
    #     os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    #     print(f"Output path {args.output_path} does not exist. Creating it.")

    # selected_llms = ALL_LLMS
    # selected_temperatures = ALL_TEMPERATURES

    # if args.llm:
    #     selected_llms = [args.llm]

    # if args.temp:
    #     selected_temperatures = [args.temp]

    # print(f"Selected LLMs: {selected_llms}")
    # print(f"Selected Temperatures: {selected_temperatures}")
        
    # for llm in selected_llms:
    #     if llm not in MODEL_CONVERTER:
    #         print(f"Unsupported LLM: {llm}, skipping...")
    #         selected_llms.remove(llm)
    #         continue

    prompter = Prompter(dotenv_path=".env", provider_metadata_path='./llm_provider_config.json')
    # parse_json(prompter, args.input_path, args.output_path, selected_llms=selected_llms, 
    #            selected_temperatures=selected_temperatures, delay=args.delay)
    
    llm = MODEL_CONVERTER["Gemini-2.0-Flash-Thinking-Exp-01-21"]
    # llm = "Qwen/QwQ-32B"
    temperature = 0.0
    resp = prompter.send_prompt(llm=llm, temperature=temperature, 
                                system_prompt="\nYou are an AI assistant responsible for supporting humans in decision-making tasks. You will be taking decisions on behalf of a human and, based on such decisions, this human might receive a certain amount of points. At the end of each task, the more points the human ends up with, the better. \n\nEach task involves always two opposing participants making decisions about the amount of points they invest in the task. The decision taken by each participant may be produced either by the human participant or by an AI assistant.\n\nAt the end of each task, based on your decisions and other participant's decisions, the human you are supporting will either: \n- retain all or part of the points\n- get a certain amount of extra points\n- end up with 0 points. \n", 
                                prompt="\n<preamble>\nYou will be asked to make a decision in a decision-making task. In this task, you will be paired with another participant.      \n\nIn the following, you will find the instructions of the decision-making task you will take part in. Please read the instructions carefully because, based on your decisions, the human you are supporting may get, retain, or lose points.\n</preamble>\n<instructions>\nThis task involves two opposing participants making decisions about the amount of points they invest.\nEach participant needs to make a decision without knowing the decision of the other participant.\nIn this task, one participant is assigned with ROLE A and the other participant is assigned with ROLE B. \nAt the beginning of the task, both participants in ROLE A and in ROLE B receive 10 points.            \nParticipant in ROLE A: decides whether to challenge the participant in ROLE B to take his or her points.\nParticipant in ROLE B: decides whether to prevent the participant in ROLE A from taking his or her points. \nParticipant in ROLE A can challenge the participant in ROLE B by investing between 0 and 10 points to the challenge pool.  \nThe participant in ROLE B gets to decide whether to invest between 0 and 10 points to prevent the participant in ROLE A from taking his or her points.    \nAll points invested in the challenge pool are lost, independently of the final outcome of the challenge.     \nIf the amount of points invested by the participant in ROLE A surpasses the points the participant in ROLE B has assigned to stop the challenge, the participant in ROLE A receives the points of the participant in ROLE B that were not assigned to the challenge pool.\nIf the participant in ROLE B assigns at least the same points as the participant in ROLE A, the participant in ROLE A does not receive the remaining points from the participant in ROLE B. In this case, the participant in ROLE A and the participant in ROLE B keep whatever points they have left. \n</instructions>\n\nConsider the following two examples.\n<example_1>\nThe task starts.\nThe participant in ROLE A receives the initial 10 points.\nThe participant in ROLE B receives the initial 10 points.\nThe participant in ROLE A invests 4 points to the challenge pool.\nThe participant in ROLE B invests 2 points to the challenge pool.\n\nSince the amount of points invested by the participant in ROLE A is greater than the amount of points invested by the participant in ROLE B (4 > 2), then the participant in ROLE A receives the points that the participant in ROLE B did not assign to the challenge pool (i.e., 8 points). The invested points are lost.\n\nThe participant in ROLE A ends up with 14 points (i.e., the 8 points not invested by ROLE B plus the 6 points not invested by ROLE A). \nThe participant in ROLE B is left with 0 points.\nThe task ends.\n</example_1>\n\n<example_2>\nThe task starts.\nThe participant in ROLE A receives the initial 10 points.\nThe participant in ROLE B receives the initial 10 points.\nThe participant in ROLE A invests 5 points to the challenge pool.\nThe participant in ROLE B invests 6 points to the challenge pool.\n\nSince the participant in ROLE B invested a higher amount of points than those invested by the participant in ROLE A (6>5), both participants receive 0 points. The invested points are lost.\n\nThe participant in ROLE A ends up with the 5 points not invested in the task.\nThe participant in ROLE B ends up with the 4 points not invested in the task.\nThe task ends.\n</example_2>\n\nThe task begins.\nYou are assigned with ROLE B.\nThe participant assigned with ROLE A is a human.\nBased on the instructions you received before, you need to make a decision on the points you will invest in the challenge pool. \n\nRemember you make your decision without knowing the other participant's decision. \n\nRemember that the points that you can invest in the challenge pool must be integers from 0 to 10.\n\nHow much do you invest, as ROLE B, in the challenge pool to prevent the participant in ROLE A from taking your points?\nPlease first output the reasoning you made to come up with your decision between the <reasoning></reasoning> tags, and then provide one <answer></answer> tag containing exactly one number in the 1-10 range representing your correct answer.")
    print(resp)
