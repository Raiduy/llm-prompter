import argparse
import json
import os

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
    parser = argparse.ArgumentParser(prog="Checkpoint Updater", description="This program can be used in the event that main.py or parallel_main.py do not write the output file.\n\
                                     It uses the input JSON file (-i) to determine the structure of what has to be added from the checkpoints folder (-c).\n \
                                     The resulting data structure is written to the output JSON file (-o)\n")
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("-c", "--checkpoints_folder", type=str, required=True, help="Path to the checkpoint folder where the checkpoint files are saved.")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Path to output JSON file.")
    args = parser.parse_args()

    IN_FILE = args.input_path
    CHECKPOINT_DIR = args.checkpoints_path
    OUT_FILE = args.output_path

    data = load_json_data(IN_FILE)
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