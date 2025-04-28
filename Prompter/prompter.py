import os
import json
from dotenv import dotenv_values

import anthropic
from google import genai
from huggingface_hub import InferenceClient
from openai import OpenAI


# Load environment variables from .env file

class Prompter:
    def __init__(self, dotenv_path, provider_metadata_path):
        self.api_keys = dotenv_values(dotenv_path)
        
        try:
            with open(provider_metadata_path, "r") as file:
                self.provider_metadata = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Provider metadata file not found: {provider_metadata_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from provider metadata file: {provider_metadata_path}")


    def prompt_openAI(self, provider, llm, temperature, system_prompt, prompt) -> str:
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

        if temperature is not None:
            api_call_parameters["temperature"] = temperature

        if llm not in self.provider_metadata[provider]["models"]:
            raise ValueError(f"Unsupported model: {llm} for provider: {provider}")
        else:
            print(f"Using {provider} API, with model: {llm}")
            response = client.chat.completions.create(**api_call_parameters)
            return response.choices[0].message.content


    def prompt_google(self, provider, llm, temperature, system_prompt, prompt) -> str:
        client = genai.Client(api_key=self.api_keys[f"{provider}_API_KEY"])
        
        config = genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        )
        
        response = client.models.generate_content(
            model=llm,
            config=config,
            contents=system_prompt,
        )

        return response.text


    def prompt_anthropic(self, provider, llm, temperature, system_prompt, prompt) -> str:
        client = anthropic.Anthropic(api_key=self.api_keys[f"{provider}_API_KEY"])

        message = client.messages.create(
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

        return message.content


    def prompt_huggingface_provider(self, provider, llm, temperature, system_prompt, prompt) -> str:
        client = InferenceClient(provider=provider, api_key=self.api_keys[f"HUGGINGFACE_API_KEY"])

        completion = client.chat.completions.create(
            model=llm,
            messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": prompt },
            ],
            temperature=temperature,
        )

        return completion.choices[0].message.content


    def send_prompt(self, llm, temperature, system_prompt, prompt) -> str:
        provider_name = ''

        for provider in self.provider_metadata:
            print(f"Checking provider: {provider}")
            if llm in self.provider_metadata[provider]["models"]:
                provider_name = provider
                print(f"Found model {llm} in provider {provider}")
                break
            
        if provider_name == '':
            raise ValueError(f"No providers found for model: {llm}")

        if self.provider_metadata[provider_name]["models"][llm]["supports_temperature"] == False:
            if temperature != 0.0:
                return f"Temperature is not supported for model {llm}"
            if temperature == 0.0:
                temperature = None
                print(f"Temperature is not supported for model {llm}. Temperature set to None.")

        if self.provider_metadata[provider_name]["prompter"] == "openAI":
            print(f"Using {provider_name} API")
            answer = self.prompt_openAI(provider_name, llm, temperature, system_prompt, prompt)
            return answer

        elif self.provider_metadata[provider_name]["prompter"] == "google":
            print(f"Using {provider_name} API")
            answer = self.prompt_google(provider_name, llm, temperature, system_prompt, prompt)
            return answer

        elif self.provider_metadata[provider_name]["prompter"] == "anthropic":
            print(f"Using {provider_name} API")
            answer = self.prompt_anthropic(provider_name, llm, temperature, system_prompt, prompt)
            return answer

        elif self.provider_metadata[provider_name]["prompter"] == "huggingface":
            print(f"Using {provider_name} API")
            hf_provider = self.provider_metadata[provider_name]["models"][llm]["hf_provider"]
            answer = self.prompt_huggingface_provider(hf_provider, llm, temperature, system_prompt, prompt)
            return answer