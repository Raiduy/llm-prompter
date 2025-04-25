import os
import json
from Prompter.prompter import Prompter
from Prompter.mappings import model_converter

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
        "o3-mini-high": "o3-mini-high",
        "o1-mini": "o1-mini",
        "DeepSeek-R1": "deepseek-reasoner",
        "DeepSeek-V3": "deepseek-chat",
}

def send_prompt(llm, temperature, format_answer, system_prompt, prompt, role=None, participant=None):
    params = {
        "llm": llm,
        "temperature": temperature,
        "format_answer": format_answer,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "role": role,
        "participant": participant
    }

    # Call the completion function from litellm
    response = completion(**params)
    
    return response


def parse_json(input_path, output_path, llms=ALL_LLMS, temperatures=ALL_TEMPERATURES):
    with open(input_path, "r") as file:
        data = json.load(file)

    system_prompt = data["system_prompt"]

    # Iterate through each LLM in the JSON data
    for llms in data["prompts"]:
        print(f"LLM: {llms['llm']}")
        llm = llms["llm"]
        for temperatures in llms["prompts"]:
            temperature = temperatures["temperature_level"]
            for cq_rep in temperatures["comprehension_questions"]:
                for cq in cq_rep["prompts"]:
                    print(f"\t\t\tQuestionID: {cq['id']}")
                    print(f"\t\t\tQuestion: {cq['prompt_type']}")
                    print(f"\t\t\tAnswer: {cq['format_answer']}")
                    print(f"\t\t\tPrompt: {cq['prompt']}")
                    if 'answer' not in cq:
                        cq["answer"] = send_prompt(
                            llm=llm,
                            temperature=temperature,
                            format_answer=cq["format_answer"],
                            system_prompt=system_prompt,
                            prompt=cq["prompt"],
                        )
                    print(f"\t\t\tAnswer: {cq['answer']}")


            for task_rep in temperatures["tasks"]:
                for task in task_rep["prompts"]:
                    print(f"\t\t\tQuestionID: {task['id']}")
                    print(f"\t\t\tRole: {task['role']}")
                    print(f"\t\t\tParticipant: {task['participant']}")
                    print(f"\t\t\tFormat Answer: {task['format_answer']}")
                    print(f"\t\t\tPrompt: {task['prompt']}")   
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
                    print(f"\t\t\tAnswer: {task['answer']}")


        ## write the modified data back to the JSON file
        if output_path == None:
            output_path = input_path
        with open(output_path, "w") as file:
            json.dump(data, file, indent=4)

if __name__ == "__main__":
    prompter = Prompter(dotenv_path=".env")
    provider = model_converter["o3-mini-high"]["provider"]
    llm = model_converter["o3-mini-high"]["model"]
    temperature = 0.7
    resp = prompter.send_prompt(provider=provider, llm=llm, temperature=temperature, 
                                system_prompt="You are a helpful assistant.", 
                                prompt="What is the capital of France?")
    print(resp)
    # Load the JSON file
    

    # # Initialize a counter for the total number of words
    # total_word_count = 0
    # # Iterate through each prompt in the JSON data
    # for prompt in data["prompts"][0]["prompts"][0]["comprehension_questions"][0]["prompts"]:
    #     # Count the number of words in the prompt
    #     total_word_count += len(prompt["prompt"].split())
    
    # for prompt in data["prompts"][0]["prompts"][0]["tasks"][0]["prompts"]:
    #     # Count the number of words in the prompt
    #     total_word_count += len(prompt["prompt"].split())

    # print(f"Total number of words in the prompts: {total_word_count}")