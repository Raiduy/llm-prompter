import os
import json
import requests
import time
from dotenv import dotenv_values

import anthropic
from google import genai
from huggingface_hub import get_inference_endpoint, list_inference_endpoints
from mistralai import Mistral
from openai import OpenAI


# Load environment variables from .env file

class Prompter:
    def __init__(self, dotenv_path, provider_metadata_path, log_folder='./logs'):
        self.api_keys = dotenv_values(dotenv_path)
        self.log_folder = log_folder
        
        try:
            with open(provider_metadata_path, "r") as file:
                self.provider_metadata = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Provider metadata file not found: {provider_metadata_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from provider metadata file: {provider_metadata_path}")

    def write_log(self, id, data):
        with open(f'{self.log_folder}/{str(id)}.log', 'w') as file:
            file.write(data)

    def prompt_openAI(self, id, provider, llm, temperature, system_prompt, prompt) -> str:
        client = OpenAI(api_key=self.api_keys[f"{provider}_API_KEY"],
                        base_url=self.provider_metadata[provider]["base_url"])

        api_call_parameters = {
            "model": llm,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        if llm == 'o1-mini':
            api_call_parameters["messages"] = [
                {"role": "user", "content": system_prompt + \
                                            '\n\n### Instruction:' + prompt + \
                                            '\n\n### Response:'},
            ]

        if temperature is not None:
            api_call_parameters["temperature"] = temperature

        if llm not in self.provider_metadata[provider]["models"]:
            raise ValueError(f"Unsupported model: {llm} for provider: {provider}")

        print(f"Using {provider} API, with model: {llm}")
        try:
            response = client.chat.completions.create(**api_call_parameters)
        except Exception as e:
            print(f"Error: {e}")
            return None

        if id:
            self.write_log(id, response)

        return response.choices[0].message.content


    def prompt_google(self, id, provider, llm, temperature, system_prompt, prompt) -> str:
        client = genai.Client(api_key=self.api_keys[f"{provider}_API_KEY"])
        
        config = genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        )
        
        try:
            response = client.models.generate_content(
                model=llm,
                config=config,
                contents=prompt,
            )
        except Exception as e:
            print(f"Google Prompter Error: {e}")
            return None
        
        if id:
            self.write_log(id, response)
        return response.text


    def prompt_anthropic(self, id, provider, llm, temperature, system_prompt, prompt) -> str:
        client = anthropic.Anthropic(api_key=self.api_keys[f"{provider}_API_KEY"])

        response = client.messages.create(
            model=llm,
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {
                    "role": "user", 
                    "content": [{ "type": "text", "text": prompt }],
                },
            ],
            temperature=temperature,
            stream=False,
        )

        if id:
            self.write_log(id, response)
        return response.content[0].text


    def prompt_huggingface_endpoint(self, id, provider, llm, temperature, system_prompt, prompt) -> str:
        client = OpenAI(
            base_url = self.provider_metadata[provider]["models"][llm]["base_url"],
            api_key = self.api_keys[f"{provider}_API_KEY"],
        )

        response = client.chat.completions.create(
          model="tgi",
          messages=[
              {
                "role": "system",
                "content": system_prompt
              },
              {
                "role": "user",
                "content": prompt
              }
          ],
          temperature=temperature,
        )

        if id:
            self.write_log(id, response)
        return response.choices[0].message.content


    def prompt_mistral(self, id, provider, llm, temperature, system_prompt, prompt) -> str:
        client = Mistral(api_key=self.api_keys[f"{provider}_API_KEY"])

        response = client.chat.complete(
            model=llm,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=temperature,
        )
 
        if id:
            self.write_log(id, response)
        return response.choices[0].message.content

    def prompt_nebula(self, id, provider, llm, temperature, system_prompt, prompt) -> str:
        url = self.provider_metadata[provider]["base_url"]
        headers = {
            'Authorization': f'Bearer {self.api_keys[f"{provider}_API_KEY"]}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": llm,
            "params": {
                "temperature": temperature,
                "num_ctx": 4096
            },
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        response = requests.post(url, headers=headers, json=data)
  
        if id:
            self.write_log(id, response)
        return response.json()['choices'][0]['message']['content']


    def send_prompt(self, id, llm, temperature, system_prompt, prompt) -> str:
        provider_name = ''

        for provider in self.provider_metadata:
            if llm in self.provider_metadata[provider]["models"]:
                provider_name = provider
                print(f"Found model {llm} in provider {provider}")
                break
            
        if provider_name == '':
            print(f"No providers found for model: {llm}. Skipping...")
            return None

        if self.provider_metadata[provider_name]["models"][llm]["supports_temperature"] == False:
            if temperature != 0.0:
                return "Temperature not supported"
            if temperature == 0.0:
                temperature = None
                print(f"Temperature is not supported for model {llm}. Using default Temperature.")

        if self.provider_metadata[provider_name]["prompter"] == "openAI":
            answer = self.prompt_openAI(id, provider_name, llm, temperature, system_prompt, prompt)
            return answer

        elif self.provider_metadata[provider_name]["prompter"] == "google":
            answer = None
            retry_limit = 5
            while answer==None and retry_limit > 0:
                if retry_limit != 5:
                    print('Request failed, sleeping 5...')
                    time.sleep(5)
                answer = self.prompt_google(id, provider_name, llm, temperature, system_prompt, prompt)
                retry_limit -= 1

            return answer

        elif self.provider_metadata[provider_name]["prompter"] == "anthropic":
            answer = self.prompt_anthropic(id, provider_name, llm, temperature, system_prompt, prompt)
            return answer

        elif self.provider_metadata[provider_name]["prompter"] == "huggingface":
            answer = self.prompt_huggingface_endpoint(id, provider_name, llm, temperature, system_prompt, prompt)
            return answer

        elif self.provider_metadata[provider_name]["prompter"] == "mistral":
            answer = self.prompt_mistral(id, provider_name, llm, temperature, system_prompt, prompt)
            return answer

        elif self.provider_metadata[provider_name]["prompter"] == "nebula":
            answer = self.prompt_nebula(id, provider_name, llm, temperature, system_prompt, prompt)
            return answer