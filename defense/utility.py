import json
import os
import random
from typing import Optional

# import g4f
from autogen import OpenAIWrapper
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
from tqdm import tqdm


def load_defense_prompt(json_path="data/prompt/defense_prompts.json"):
    with open(json_path, 'r') as f:
        defense_prompt = json.load(f)
    return defense_prompt


def load_llm_config(json_path="data/config/llm_config_list.json", model_name="gpt-35-turbo",
                    port=8000, host_name="127.0.0.1", cache_seed=123,
                    frequency_penalty=0.0, temperature=0.7, presence_penalty=0.0, max_tokens=400,
                    api_key: Optional[str] = None, base_url: Optional[str] = None):
    """
    Load LLM configuration for connecting to a vLLM server or OpenAI API.
    
    Args:
        json_path: Path to an optional config JSON file. If missing, we fall back to env/CLI.
        model_name: Name of the model to use
        port: Port number where vLLM server is running (default: 8000)
        host_name: Hostname of the vLLM server (default: 127.0.0.1)
        cache_seed: Cache seed for reproducibility
        frequency_penalty: Frequency penalty for generation
        temperature: Temperature for generation
        presence_penalty: Presence penalty for generation
        max_tokens: Maximum tokens to generate
        api_key: Optional API key override (otherwise uses OPENAI_API_KEY when needed)
        base_url: Optional base URL override (otherwise uses OPENAI_BASE_URL when set)
    
    Returns:
        List of config dictionaries for the LLM
    """
    host_name = host_name if type(host_name) is str else random.choice(host_name)

    data = None
    try:
        with open(json_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = None
    except json.JSONDecodeError:
        data = None

    if isinstance(data, list):
        config = [d for d in data if d.get("model") == model_name]
        if model_name == "gpt-4":
            for c in config:
                c["temperature"] = temperature
        else:
            for c in config:
                c["cache_seed"] = cache_seed
                c["frequency_penalty"] = frequency_penalty
                c["presence_penalty"] = presence_penalty
                c["temperature"] = temperature
                c["max_tokens"] = max_tokens
        if len(config) > 0:
            return config

    # Fallback without config file:
    # - If OPENAI_API_KEY is present (or api_key passed) and this looks like an OpenAI model,
    #   use OpenAI (optionally OPENAI_BASE_URL).
    # - Otherwise, assume a local vLLM OpenAI-compatible server at host:port.
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
    model_lower = (model_name or "").lower()
    looks_like_openai = model_lower.startswith(("gpt", "o", "chatgpt"))

    if (resolved_api_key and looks_like_openai) or resolved_base_url:
        cfg = {
            "model": model_name,
            "api_key": resolved_api_key or "EMPTY",
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "cache_seed": cache_seed,
            "max_tokens": max_tokens,
        }
        if resolved_base_url:
            cfg["base_url"] = resolved_base_url
        return [cfg]

    # Default: local vLLM server
    return [{
        "model": model_name,
        "api_key": "EMPTY",  # vLLM doesn't require an API key by default
        "base_url": f"http://{host_name}:{port}/v1/",
        "temperature": temperature,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "cache_seed": cache_seed,
        "max_tokens": max_tokens,
    }]


def load_attack_template(json_path="data/prompt/attack_prompt_template.json", template_name='v1'):
    with open(json_path) as f:
        data = json.load(f)
        return [d for d in data if d['name'] == template_name][0]['template']


def load_harmful_prompt(json_path="data/prompt/prompts_curated.json", exclude_list=("unicorn", "favorite_movie")):
    with open(json_path) as f:
        data = json.load(f)
        return {k: v for k, v in data.items() if k not in exclude_list}


class G4FWrapper(OpenAIWrapper):
    def _client(self, config, openai_config):
        """Create a client with the given config to overrdie openai_config,
        after removing extra kwargs.
        """
        return g4f

    def _completions_create(self, client, params):
        completions = client.ChatCompletion
        # If streaming is not enabled or using functions, send a regular chat completion request
        # Functions are not supported, so ensure streaming is disabled
        params = params.copy()
        params["stream"] = False
        message = completions.create(**params)

        response = ChatCompletion(
            id="g4f",
            created=1,
            model=params["model"],
            object="chat.completion",
            choices=[Choice(index=0, finish_reason="stop", logprobs=None,
                            message=ChatCompletionMessage(role="assistant", content=message, function_call=None))],
            usage=CompletionUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )
        )
        return response


if __name__ == '__main__':
    # Example: Connect to a vLLM server running on localhost:8000
    # vLLM handles load balancing internally via --data-parallel-size
    llm = OpenAIWrapper(config_list=load_llm_config(model_name="llama-2-70b", port=8000),
                        cache_seed=None)

    from joblib import Parallel, delayed

    # vLLM server can handle concurrent requests efficiently
    output = (Parallel(n_jobs=24, backend='threading')
              (delayed(llm.create)
                   (messages=[
                   {'role': 'system', 'content': "You are a helpful assistant."},
                   {'role': 'user', 'content': "Hi, Please compute the sum of 1 and 2."}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
               for _ in tqdm(range(1000))))

    print([i.choices[0].message.content for i in output])
