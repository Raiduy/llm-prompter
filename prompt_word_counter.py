import json

if __name__ == "__main__":
    # Load the JSON file
    with open("prompts_plan.json", "r") as file:
        data = json.load(file)


    all_llms = set()
    all_temperatures = set()
    for prompt in data["prompts"]:
        all_llms.add(prompt["llm"])
        for temperature in prompt["prompts"]:
            all_temperatures.add(temperature["temperature_level"])
    
    print("All LLMs:", all_llms)
    print("All Temperatures:", all_temperatures)
        