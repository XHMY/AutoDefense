import openai
from autogen import OpenAIWrapper
import json
from tqdm import tqdm
from defense.utility import load_llm_config, load_attack_template, load_harmful_prompt
from joblib import Parallel, delayed


def attack_llm_collect_response(template, prompts, llm_config_list, model=None):
    llm = OpenAIWrapper(config_list=llm_config_list)
    llm_backup = OpenAIWrapper(config_list=load_llm_config(model_name="gpt-3.5-turbo-1106",
                                                           cache_seed=llm_config_list[0]["cache_seed"],
                                                           temperature=1.0))
    outputs = []
    non_success_cnt = 0
    for k, prompt in tqdm(prompts.items()):
        try:
            llm_raw_response = llm.create(model=model,
                                          messages=[{'role': 'user', 'content':
                                              template.replace("[INSERT PROMPT HERE]", prompt)}])
            assert llm_raw_response.choices[0].finish_reason != 'content_filter'
            content = llm_raw_response.choices[0].message.content
        except Exception as e:
            print("Azure API failed, using backup OpenAI model")
            content = llm_backup.create(model=model, messages=[{'role': 'user',
                                                                'content': template.replace("[INSERT PROMPT HERE]", prompt)}]).choices[0].message.content
        outputs.append({"name": k, "raw_response": content.strip()})
    return outputs, non_success_cnt


def attack(output_prefix="attack", model_name="gpt-3.5-turbo-1106", output_suffix="", cache_seed=123,
           port_range=(9005, 9005), host_name="127.0.0.1", template=None, prompts=None):
    llm_config_list = load_llm_config(model_name=model_name, cache_seed=cache_seed, temperature=1.0,
                                      port_range=port_range, host_name=host_name, presence_penalty=0.0,
                                      frequency_penalty=0.0)
    outputs, non_success_cnt = attack_llm_collect_response(template, prompts, llm_config_list)
    with open(f"data/harmful_output/{model_name}/{output_prefix}_{output_suffix}.json", "w") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
    print("non_success_cnt:", non_success_cnt,
          "total:", len(prompts),
          "non-success rate:", non_success_cnt / len(prompts))


if __name__ == '__main__':
    model_name = "gpt-35-turbo-1106"
    port_range = (9005, 9005 + 1)
    host_name = "dgxh-1"

    # get curated + synthetic attack output
    # prompts_synthetic = load_harmful_prompt(json_path="data/prompt/prompts_synthetic.json")
    # prompts = {**prompts_curated, **prompts_synthetic}

    (Parallel(n_jobs=2, backend='threading')
     (delayed(attack)(output_prefix="attack-dan", output_suffix=f"{i}", cache_seed=i, model_name=model_name,
                      port_range=port_range, host_name=host_name,
                      template=load_attack_template(template_name="v1"),
                      prompts=load_harmful_prompt(json_path="data/prompt/prompt_dan.json"))
      for i in range(5)))

    # get safe output
    # (Parallel(n_jobs=2, backend='threading')
    #  (delayed(attack)(output_prefix="safe", output_suffix=f"{i}", cache_seed=i, model_name=model_name,
    #                   port_range=port_range, host_name=host_name,
    #                   template=load_attack_template(template_name="placeholder"),
    #                   prompts=load_harmful_prompt(json_path="data/prompt/safe_prompts.json"))
    #   for i in range(5)))


