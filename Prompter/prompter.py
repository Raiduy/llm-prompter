import os
import json
from dotenv import dotenv_values
from openai import OpenAI
from Prompter.mappings import provider_metadata


# Load environment variables from .env file

class Prompter:
    def __init__(self, dotenv_path):
        self.api_keys = dotenv_values(dotenv_path)

    def prompt_openAI(self, provider, llm, temperature, system_prompt, prompt) -> str:
        client = OpenAI(api_key=self.api_keys[f"{provider}_API_KEY"],
                        base_url=provider_metadata[provider]["base_url"])
        print(f"Using {provider} API, with model: {llm}")
        if llm not in provider_metadata[provider]["models"]:
            raise ValueError(f"Unsupported model: {llm} for provider: {provider}")
        else:
            response = client.chat.completions.create(
                model=llm,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=temperature
            )
            
            return response.choices[0].message.content


    def send_prompt(self, provider, llm, temperature, system_prompt, prompt) -> str:
        # if llm == "gpt-3.5-turbo":
        #     response = openai.ChatCompletion.create(
        #         model=llm,
        #         messages=[
        #             {"role": "system", "content": system_prompt},
        #             {"role": "user", "content": prompt}
        #         ],
        #         temperature=temperature,
        #     )
        #     answer = response.choices[0].message.content.strip()
        # elif llm == "gpt-4":
        #     response = openai.ChatCompletion.create(
        #         model=llm,
        #         messages=[
        #             {"role": "system", "content": system_prompt},
        #             {"role": "user", "content": prompt}
        #         ],
        #         temperature=temperature,
        #     )
        #     answer = response.choices[0].message.content.strip()
        # else:
        #     raise ValueError(f"Unsupported LLM: {llm}")
        if provider_metadata[provider]["prompter"] == "openAI":
            print("Using OpenAI API")
            answer = self.prompt_openAI(provider, llm, temperature, system_prompt, prompt)
        return answer