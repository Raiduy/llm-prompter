import json
import os
import re

CHECKPOINT_DIR = './pll_checkpoints'
OUT_FILE = 'pls_worc.json'


def load_json_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def parse_checkpoint_file(filename):
    filepath = f'{CHECKPOINT_DIR}/{filename}'


    filename = filename.split('.json')[0]
    model = filename.split('_temp_')[0]
    temperature = filename.split('_temp_')[1].split('_rep_')[0]
    repetition = filename.split('_rep_')[1].split('_')[0]
    question_type = filename.split('_')[-1]

    checkpoint_data = load_json_data(filepath)

    return model, temperature, repetition, question_type, checkpoint_data



if __name__=='__main__':
    with open('./prompts_plan_FULL_1_REP_NEW_ANSWERS.json', 'r') as f:
        data = json.load(f)

    cp_files = os.listdir(CHECKPOINT_DIR)
        
    for i, llms in enumerate(data["prompts"]):
        llm = llms["llm"]
        if f'{llm}.json' in cp_files:
            data["prompts"][i] = load_json_data(f'{CHECKPOINT_DIR}/{llm}.json')
        elif any(f'{llm}' in file for file in cp_files):
            for j, temperatures in enumerate(llms["prompts"]):
                temperature = temperatures["temperature_level"]
                for k, cq_reps in enumerate(temperatures["comprehension_questions"]):
                    cq_rep = cq_reps["repetition_id"]
                    if os.path.exists(f'{CHECKPOINT_DIR}/{llm}_temp_{temperature}_rep_{cq_rep}_cq.json'):
                        data["prompts"][i]["prompts"][j]["comprehension_questions"][k] = load_json_data(f'{CHECKPOINT_DIR}/{llm}_temp_{temperature}_rep_{cq_rep}_cq.json')
                for k, task_reps in enumerate(temperatures["tasks"]):
                    task_rep = task_reps["repetition_id"]
                    if os.path.exists(f'{CHECKPOINT_DIR}/{llm}_temp_{temperature}_rep_{task_rep}_task.json'):
                        data["prompts"][i]["prompts"][j]["tasks"][k] = load_json_data(f'{CHECKPOINT_DIR}/{llm}_temp_{temperature}_rep_{task_rep}_task.json')
    with open(OUT_FILE, 'w') as out:
        json.dump(data, out, indent=4)