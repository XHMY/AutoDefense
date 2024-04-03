import json
import random

# import g4f
# import httpx
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
                    port_range=(9005, 9005), host_name="127.0.0.1", cache_seed=123,
                    frequency_penalty=0.0, temperature=0.7, presence_penalty=0.0, max_tokens=400):
    host_name = host_name if type(host_name) is str else random.choice(host_name)
    with open(json_path) as f:
        data = json.load(f)
        config = [d for d in data if d['model'] == model_name]
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
        else:
            return [{
                "model": model_name,
                "api_key": "AAA",
                "base_url": f"http://{host_name}:{random.randint(*port_range)}/v1/",
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


class LoadBalanceLlamaWrapper(OpenAIWrapper):
    def __init__(self, *args, **kwargs):
        self.port_range = kwargs.pop("port_range", (9005, 9005))
        super().__init__(*args, **kwargs)

    def _completions_create(self, client, params):
        client.base_url = httpx.URL(client.base_url, port=random.randint(*self.port_range))
        # print("Select port:", client.base_url)
        response = super()._completions_create(client, params)
        return response


if __name__ == '__main__':
    llm = LoadBalanceLlamaWrapper(config_list=load_llm_config(model_name="llama-2-70b"),
                                  cache_seed=None, port_range=(9005, 9005 + 3))

    # llm.create(messages=[{'role': 'user', 'content': "hello"}], model="llama-2-13b")

    from joblib import Parallel, delayed

    output = (Parallel(n_jobs=24, backend='threading')
              (delayed(llm.create)
                   (messages=[
                   {'role': 'system', 'content': "You are a helpful assistant."},
                   {'role': 'user', 'content': "Hi, Please compute the sum of 1 and 2."}])
               for _ in tqdm(range(1000))))

    print([i.choices[0].message.content for i in output])
