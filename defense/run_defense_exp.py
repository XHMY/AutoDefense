import os
from os.path import join, exists
from joblib import Parallel, delayed
import pandas as pd
from glob import glob
from tqdm import tqdm
from defense.explicit_detector.agency.explicit_1_agent import (VanillaJailbreakDetector, CoT, CoTV2, CoTV3,
                                                               VanillaJailbreakDetectorV0125)
from defense.explicit_detector.agency.explicit_2_agents import AutoGenDetectorV1, AutoGenDetectorV0125
from defense.explicit_detector.agency.explicit_3_agents import AutoGenDetectorThreeAgency, AutoGenDetectorThreeAgencyV2
from defense.explicit_detector.agency.explicit_5_agents import DetectorFiveAgency
from defense.explicit_detector.explicit_defense_arch import ExplicitMultiAgentDefense
# from defense.implicit_detector.agency.implicit_1_agent import MoralAdvisor
# from defense.implicit_detector.agency.implicit_2_agents import MoralAdvisor2Agent
# from defense.implicit_detector.agency.implicit_3_agents import MoralAdvisor3Agent
# from defense.implicit_detector.implicit_defense_arch import ImplicitMultiAgentDefense
from evaluator.evaluate_helper import evaluate_defense_with_output_list, evaluate_defense_with_response
import argparse

defense_strategies = [
    # {"name": "im-1", "defense_agency": ImplicitMultiAgentDefense, "task_agency": MoralAdvisor},
    # {"name": "im-2", "defense_agency": ImplicitMultiAgentDefense, "task_agency": MoralAdvisor2Agent},
    # {"name": "im-3", "defense_agency": ImplicitMultiAgentDefense, "task_agency": MoralAdvisor3Agent},
    # {"name": "ex-1", "defense_agency": ExplicitMultiAgentDefense, "task_agency": VanillaJailbreakDetector},
    {"name": "ex-2", "defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorV1},
    {"name": "ex-3", "defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorThreeAgency},
    {"name": "ex-cot", "defense_agency": ExplicitMultiAgentDefense, "task_agency": CoT},
    # {"name": "ex-1-0125", "defense_agency": ExplicitMultiAgentDefense, "task_agency": VanillaJailbreakDetectorV0125},
    # {"name": "ex-2-0125", "defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorV0125},
    # {"name": "ex-cot-5", "defense_agency": ExplicitMultiAgentDefense, "task_agency": CoTV3},
    # {"name": "ex-5", "defense_agency": ExplicitMultiAgentDefense, "task_agency": DetectorFiveAgency},
    # {"name": "ex-3-v2", "defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorThreeAgencyV2},
    # {"name": "ex-cot-v2", "defense_agency": ExplicitMultiAgentDefense, "task_agency": CoTV2},
]


def eval_csv_from_yuan():
    attack_csv_list = glob("data/harmful_output/multiple_attack_output/*.csv")
    attack_csv_list.sort()
    for attack_csv in tqdm(attack_csv_list):
        df = pd.read_csv(attack_csv)
        for defense_strategy in defense_strategies:
            df[defense_strategy["name"]] = evaluate_defense_with_output_list(
                task_agency=defense_strategy["task_agency"],
                defense_agency=defense_strategy["defense_agency"],
                output_list=df["target"].tolist())

        df.to_csv(attack_csv, index=False)


def eval_defense_strategies(llm_name, output_suffix, ignore_existing=True,
                            chat_file="data/harmful_output/attack_gpt3.5_1106.json",
                            host_name="127.0.0.1", port_range=(9005, 9005),
                            frequency_penalty=1.3, num_of_threads=6, temperature=0.7, presence_penalty=0.0):
    defense_output_prefix = join(f"data/defense_output/open-llm-defense{output_suffix}", llm_name)
    os.makedirs(defense_output_prefix, exist_ok=True)
    for defense_strategy in defense_strategies:
        output_file = join(defense_output_prefix, defense_strategy["name"] + ".json")
        if exists(output_file) and ignore_existing:
            print("Defense output exists, skip", output_file)
            continue
        print("Evaluating", llm_name, defense_strategy["name"], "\nOutput to", output_file)
        evaluate_defense_with_response(
            task_agency=defense_strategy["task_agency"],
            defense_agency=defense_strategy["defense_agency"],
            chat_file=chat_file,
            defense_output_name=join(defense_output_prefix, defense_strategy["name"] + ".json"),
            model_name=llm_name,
            port_range=port_range,
            host_name=host_name,
            parallel=True, num_of_threads=num_of_threads,
            frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
            temperature=temperature)


def eval_with_open_llms(model_list, chat_file, port_range=(9005, 9005 + 3), ignore_existing=True,
                        host_name="127.0.0.1", output_suffix="", frequency_penalty=1.3,
                        temperature=0.7, eval_safe=True, eval_harm=True, presence_penalty=0.0):
    # "llama-2-13b", "llama-2-7b", "llama-pro-8b", "llama-2-70b", "tinyllama-1.1b", "vicuna-13b-v1.5", "vicuna-33b", "vicuna-7b-v1.5", "vicuna-13b-v1.3.0"
    for llm_name in model_list:
        print("Evaluating", llm_name)
        if eval_harm:
            eval_defense_strategies(llm_name, output_suffix, ignore_existing=ignore_existing,
                                    chat_file=chat_file,
                                    host_name=host_name, port_range=port_range, presence_penalty=presence_penalty,
                                    frequency_penalty=frequency_penalty, temperature=temperature)
        if eval_safe:
            eval_defense_strategies(llm_name, "-safe" + output_suffix, ignore_existing=ignore_existing,
                                    chat_file=chat_file.replace("attack", "safe"),
                                    host_name=host_name, port_range=port_range, presence_penalty=presence_penalty,
                                    frequency_penalty=frequency_penalty, temperature=temperature)


def eval_with_openai(model_list, chat_file, ignore_existing=True, output_suffix="",
                     temperature=0.7, eval_safe=True, eval_harm=True, presence_penalty=0.0):
    for llm_name in model_list:
        print("Evaluating", llm_name)
        if eval_harm:
            eval_defense_strategies(llm_name, output_suffix, ignore_existing=ignore_existing,
                                    chat_file=chat_file, presence_penalty=presence_penalty,
                                    num_of_threads=2, temperature=temperature)
        if eval_safe:
            eval_defense_strategies(llm_name, "-safe" + output_suffix, ignore_existing=ignore_existing,
                                    chat_file=chat_file.replace("attack_gpt3.5", "safe_gpt3.5"),
                                    presence_penalty=presence_penalty,
                                    num_of_threads=2, temperature=temperature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_list", nargs="*", default=["gpt-3.5-turbo-1106"])
    parser.add_argument("--chat_file", type=str, default="data/harmful_output/attack_gpt3.5_1106.json")
    parser.add_argument("--port_start", type=int, default=9005)
    parser.add_argument("--host_name", nargs="*", default=["127.0.0.1"])
    parser.add_argument("--num_of_instance", type=int, default=1)
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--eval_harm", action="store_true")
    parser.add_argument("--eval_safe", action="store_true")
    args = parser.parse_args()

    if args.model_list[0].startswith("gpt"):
        eval_with_openai(model_list=args.model_list, output_suffix=args.output_suffix, ignore_existing=True,
                         temperature=args.temperature, chat_file=args.chat_file, eval_safe=args.eval_safe,
                         eval_harm=args.eval_harm, presence_penalty=args.presence_penalty)
    else:
        port_range = (args.port_start, args.port_start + args.num_of_instance - 1)
        eval_with_open_llms(model_list=args.model_list, port_range=port_range, ignore_existing=True,
                            output_suffix=args.output_suffix, host_name=args.host_name,
                            frequency_penalty=args.frequency_penalty, temperature=args.temperature,
                            chat_file=args.chat_file, eval_safe=args.eval_safe, eval_harm=args.eval_harm,
                            presence_penalty=args.presence_penalty)
