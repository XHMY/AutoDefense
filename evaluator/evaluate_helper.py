import json
import os
from glob import glob

import openai
import pandas as pd
from joblib import Parallel, delayed
from retry import retry
from tqdm import tqdm

from defense.utility import load_llm_config, load_attack_template, load_harmful_prompt


def evaluate_explicit_detector(detector, log_file, attack_output_file="data/harmful_output/attack_gpt3.5_1106.json"):
    statistics_list = []
    for chat_file in glob(attack_output_file):
        if "summary" in chat_file or "draft" in chat_file or "detection" in chat_file:
            continue

        print("Processing", chat_file)

        with open(chat_file) as f:
            data = json.load(f)
            harmful_response = [[k, v] for k, v in data if v != "ERROR"]

        if len(harmful_response) == 0:
            continue

        results = []
        for k, v in tqdm(harmful_response):
            valid, success, llm_detection_log_output, final_output = detector(v)
            if not success:
                print("Fail content:", v["response"])
                print("Fail output:", llm_detection_log_output)
            results.append({
                "name": k,
                "input": v,
                "valid": valid,
                "success": success,
                "llm_detection_log_output": llm_detection_log_output,
                "final_output": final_output
            })

        statistics_list.append({
            "chat_file": chat_file,
            "statistics": {
                "total": len(results),
                "valid": sum([1 for r in results if r["valid"]]),
                "invalid": sum([1 for r in results if not r["valid"]]),
                "success": sum([1 for r in results if r["success"]]),
                "fail": sum([1 for r in results if not r["success"]]),
                "invalid_rate": sum([1 for r in results if not r["valid"]]) / len(results),
            },
            "results": results})

        print("Statistics:", statistics_list[-1]["statistics"])

    with open(log_file, "w") as f:
        json.dump(statistics_list, f, indent=4, ensure_ascii=False)


def evaluate_defense_with_response(task_agency, defense_agency, defense_output_name,
                                   chat_file="data/harmful_output/attack_gpt3.5_1106.json",
                                   model_name="gpt-35-turbo", port_range=(9005, 9005), parallel=False,
                                   host_name="127.0.0.1", num_of_repetition=5, frequency_penalty=1.3,
                                   num_of_threads=6, temperature=0.7, cache_seed=123, presence_penalty=0.0):
    harmful_response = []
    if os.path.exists(chat_file):
        ignore_num_of_rep = False
        for rep_idx in range(num_of_repetition):
            with open(chat_file) as f:
                data = json.load(f)
                harmful_response += [[f"{i['name']}-{rep_idx}", i["raw_response"]] for i in data
                                     if i["raw_response"] != "ERROR"
                                     and i["name"] not in ("unicorn", "favorite_movie")]
    else:
        file_list = glob(chat_file.replace(".json", "*.json"))
        assert len(file_list) > 0, f'Cannot find any chat file in {chat_file.replace(".json", "*.json")}'
        ignore_num_of_rep = True
        print("Use separate chat files, ignore num_of_repetition=", num_of_repetition)
        for cf in file_list:
            with open(cf) as f:
                data = json.load(f)
                harmful_response += [[f"{i['name']}-{cf.split('_')[-1].split('.')[0]}", i["raw_response"]] for i in data
                                     if i["raw_response"] != "ERROR"
                                     and i["name"] not in ("unicorn", "favorite_movie")]

    if parallel:
        @retry(openai.RateLimitError, tries=20, delay=10, backoff=20)
        def f(k, v, cache_seed, frequency_penalty=0.0, presence_penalty=0.0):
            llm_config = load_llm_config(model_name=model_name, port_range=port_range,
                                         host_name=host_name, cache_seed=cache_seed,
                                         frequency_penalty=frequency_penalty, temperature=temperature,
                                         presence_penalty=presence_penalty)
            out = defense_agency(task_agency=task_agency(
                config_list=llm_config), config_list=llm_config).defense_with_response(response=v)["content"]
            return {"name": k, "raw_response": v, "defense_response": out}

        # defense_output = Parallel(n_jobs=1, backend='sequential')( # for debug
        defense_output = Parallel(n_jobs=(port_range[1] - port_range[0] + 1) * num_of_threads *
                                         (len(host_name) if type(host_name) is list else 1),
                                  backend='threading')(
            delayed(f)(k, v, cache_seed if ignore_num_of_rep else int(k.split('-')[-1]), frequency_penalty, presence_penalty)
            for k, v in tqdm(harmful_response))
    else:
        defense_output = []
        for k, v in tqdm(harmful_response):
            llm_config = load_llm_config(model_name=model_name, port_range=port_range,
                                         frequency_penalty=frequency_penalty, temperature=temperature,
                                         host_name=host_name, cache_seed=cache_seed if ignore_num_of_rep else int(k.split('-')[-1]),
                                         presence_penalty=presence_penalty
            )
            defense = defense_agency(task_agency=task_agency(config_list=llm_config),
                                     config_list=llm_config)
            final_output = defense.defense_with_response(v)["content"]
            defense_output.append({"name": k, "raw_response": v, "defense_response": final_output})

    with open(defense_output_name, "w") as f:
        json.dump(defense_output, f, indent=4, ensure_ascii=False)


def evaluate_defense_with_output_list(task_agency, defense_agency, output_list, model_name="gpt-35-turbo"):
    llm_config = load_llm_config(model_name=model_name)
    defense = defense_agency(task_agency=task_agency(),
                             config_list=llm_config)

    return [defense.defense_with_response(harmful_response)["content"] for harmful_response in tqdm(output_list)]


def evaluate_defense_with_prompt(task_agency, defense_agency, defense_output_name):
    llm_config = load_llm_config()
    defense = defense_agency(task_agency=task_agency(),
                             config_list=llm_config)
    template = load_attack_template()
    prompts = load_harmful_prompt()
    defense_output = []
    for k, prompt in tqdm(prompts.items()):
        final_output = defense.defense_with_prompt(template.replace("[INSERT PROMPT HERE]", prompt))["content"]
        defense_output.append({"name": k,
                               "raw_response": defense.taskagency_agent.last_message()["content"],
                               "defense_response": final_output})

    with open(f"data/defense_output/{defense_output_name}", "w") as f:
        json.dump(defense_output, f, indent=4, ensure_ascii=False)


def eval_defense_output(cal_asr_func, output_name="asr.csv"):
    # exclude_list = (
    #     "attractiveness_stereotypes",
    #     "favorite_movie",
    #     "bullying_husbands_mistress",
    #     "illegal_firearms",
    #     "hate_speech",
    #     "maximizing_pain"
    # )
    exclude_list = None

    asr_results = []
    for defense_output_file in glob("data/defense_output/*.json"):
        with open(defense_output_file, 'r') as f:
            defense_output = json.load(f)
        asr_res = cal_asr_func(defense_output, exclude_list=exclude_list)
        asr_results.append({"Exp Name": defense_output_file.split(os.sep)[-1].split('.')[0],
                            "asr": asr_res})

    asr_res = cal_asr_func(defense_output, eval_field="raw_response", exclude_list=exclude_list)
    asr_results.append({"Exp Name": "Baseline", "asr": asr_res})

    asr_df = pd.DataFrame(asr_results).sort_values(by="asr", ascending=True)
    asr_df.to_csv(f"data/defense_output/{output_name}", index=False)
    print(asr_df)
    return asr_df
