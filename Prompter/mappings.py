model_converter = {
        "GPT-4.5-Preview": {
            "model": "gpt-4.5-preview-2025-02-27",
            "provider": "OPENAI",
        },
        "ChatGPT-4o-latest (2025-01-29)": {
            "model": "chatgpt-4o-latest",
            "provider": "OPENAI",
        },
        "o1-2024-12-17": {
            "model": "o1-2024-12-17",
            "provider": "OPENAI",
        },
        "o3-mini-high": {
            "model": "o3-mini",
            "provider": "OPENAI",
        },
        "o1-mini": {
            "model": "o1-mini",
            "provider": "OPENAI",
        },
        "DeepSeek-R1": {
            "model": "deepseek-reasoner",
            "provider": "DEEPSEEK",
        },
        "DeepSeek-V3": {
            "model": "deepseek-chat",
            "provider": "DEEPSEEK",
        },

}

provider_metadata = {
    "OPENAI": {
        "base_url": "https://api.openai.com/v1",
        "prompter": "openAI",
        "models": ["gpt-4.5-preview-2025-02-27", "chatgpt-4o-latest", "o1-2024-12-17", "o3-mini", "o1-mini"],
    },
    "DEEPSEEK": {
        "base_url": "https://api.deepseek.com",
        "prompter": "openAI",
        "models": ["deepseek-reasoner", "deepseek-chat"],
    },
}