import json
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/LlamaGuard-7b"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

llamaguard_exp_config = [
    {"output_name": "apr", "template_name": "v1", "chat_file": "gpt-35-turbo-1106/attack-dan.json"},
    {"output_name": "pr", "template_name": "placeholder", "chat_file": "gpt-35-turbo-1106/attack-dan.json"},
    {"output_name": "r", "template_name": "empty", "chat_file": "gpt-35-turbo-1106/attack-dan.json"},
    # {"output_name": "safe-apr", "template_name": "v1", "chat_file": "alpaca_data_safe_1000.json"},
    # {"output_name": "safe-pr", "template_name": "placeholder", "chat_file": "alpaca_data_safe_1000.json"},
    # {"output_name": "safe-r", "template_name": "empty", "chat_file": "alpaca_data_safe_1000.json"}
]


def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


def load_harmful_prompt(json_path="../../data/prompt/prompt_dan.json", exclude_list=("unicorn", "favorite_movie")):
    with open(json_path) as f:
        data = json.load(f)
        return {k: v for k, v in data.items() if k not in exclude_list}


def load_attack_template(json_path="../../data/prompt/attack_prompt_template.json", template_name='v1'):
    with open(json_path) as f:
        data = json.load(f)
        return [d for d in data if d['name'] == template_name][0]['template']


def run_experiment(chat_file="../../data/harmful_output/attack_gpt3.5_1106.json", template_name="v1", output_name="pr"):
    template = load_attack_template(template_name=template_name)
    harmful_entries = []
    harmful_prompt = load_harmful_prompt()
    safe_prompt = load_harmful_prompt(json_path="../../data/prompt/safe_prompts.json")
    all_prompts = {**harmful_prompt, **safe_prompt}

    for cf in glob(chat_file.replace(".json", "*.json")):
        with open(cf) as f:
            data = json.load(f)
            harmful_entries += [
                [f"{i['name']}-{cf.split('_')[-1].split('.')[0]}", i["prompt"] if "prompt" in i.keys() else all_prompts[i['name']], i["raw_response"]] for i
                in data if
                i["raw_response"] != "ERROR" and (i["name"] in all_prompts.keys() or "prompt" in i.keys())]

    defense_output = []
    for name, prompt, response in tqdm(harmful_entries):
        defense_output.append({"name": name,
                               "prompt": prompt,
                               "raw_response": response,
                               "guard_response": moderate([
                                   {"role": "user", "content": template.replace("[INSERT PROMPT HERE]", prompt)},
                                   {"role": "assistant", "content": response}])})

    with open(f"../../data/defense_output/compared_baseline/llamaguard/{output_name}.json", "w", encoding="utf-8") as f:
        json.dump(post_process(defense_output), f, indent=4, ensure_ascii=False)

    return defense_output


def post_process(defense_output):
    for idx, i in enumerate(defense_output):
        if i["guard_response"].lower().startswith("safe"):
            defense_output[idx]["defense_response"] = i["raw_response"]
        elif i["guard_response"].lower().startswith("unsafe"):
            defense_output[idx][
                "defense_response"] = "I'm sorry, but I'm not able to provide the information you're looking for."
        else:
            defense_output[idx]["defense_response"] = "DEFENSE ERROR: response not begin with safe or unsafe"
    return defense_output


if __name__ == '__main__':
    for exp in llamaguard_exp_config:
        run_experiment(chat_file=f"../../data/harmful_output/{exp['chat_file']}",
                       template_name=exp['template_name'],
                       output_name=exp['output_name'])
