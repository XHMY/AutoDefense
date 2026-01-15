import openai
from autogen import OpenAIWrapper
from defense.explicit_detector.explicit_defense_arch import ExplicitMultiAgentDefense
from defense.utility import load_defense_prompt
from evaluator.evaluate_helper import evaluate_defense_with_response


class VanillaJailbreakDetector:
    def __init__(self, config_list):
        self.defense_prompt = load_defense_prompt()
        self.strategy_name = "explicit_1_agent"
        self.llm = OpenAIWrapper(config_list=config_list, cache_seed=123)
        self.messages = [
            {
                "role": "system",
                "content": self.defense_prompt[self.strategy_name]["1_system"],
            }
        ]

    def __call__(self, llm_output, user_prompt=None):
        assert user_prompt is None

        messages = self.messages + [
            {
                "role": "user",
                "content": self.defense_prompt[self.strategy_name]["1_user"]
                .replace("[INSERT INPUT HERE]", llm_output),
            }
        ]

        try:
            llm_output = self.llm.create(messages=messages,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                ).choices[0].message.content
        except openai.BadRequestError as e:
            llm_output = "INVALID"

        if llm_output is None:
            llm_output = ""

        success = True
        if "INVALID" in llm_output.upper():
            valid = False
        elif "VALID" in llm_output.upper():
            valid = True
        else:
            valid = False
            success = False

        return valid, success, llm_output

    def agency_execute(self, llm_output):
        valid, success, llm_output = self(llm_output)
        return llm_output


class CoT(VanillaJailbreakDetector):
    def __init__(self, config_list):
        self.defense_prompt = load_defense_prompt()
        self.strategy_name = "explicit_1_agent_CoT"
        self.llm = OpenAIWrapper(config_list=config_list, cache_seed=123)
        self.messages = [
            {
                "role": "system",
                "content": self.defense_prompt[self.strategy_name]["1_system"],
            }
        ]


class CoTV2(VanillaJailbreakDetector):
    def __init__(self, config_list):
        self.defense_prompt = load_defense_prompt()
        self.strategy_name = "explicit_1_agent_CoT_v2"
        self.llm = OpenAIWrapper(config_list=config_list, cache_seed=123)
        self.messages = [
            {
                "role": "system",
                "content": self.defense_prompt[self.strategy_name]["1_system"],
            }
        ]

class CoTV3(VanillaJailbreakDetector):
    def __init__(self, config_list):
        self.defense_prompt = load_defense_prompt()
        self.strategy_name = "explicit_1_agent_CoT_v3"
        self.llm = OpenAIWrapper(config_list=config_list, cache_seed=123)
        self.messages = [
            {
                "role": "system",
                "content": self.defense_prompt[self.strategy_name]["1_system"],
            }
        ]

class VanillaJailbreakDetectorV0125(VanillaJailbreakDetector):
    def __init__(self, config_list):
        self.defense_prompt = load_defense_prompt()
        self.strategy_name = "explicit_1_agent_v0125"
        self.llm = OpenAIWrapper(config_list=config_list, cache_seed=123)
        self.messages = [
            {
                "role": "system",
                "content": self.defense_prompt[self.strategy_name]["1_system"],
            }
        ]

if __name__ == '__main__':
    # args = argparse.ArgumentParser()
    # args.add_argument("--log_file", type=str, default="data/defense_output/detection_summary_vanilla.json")
    # args = args.parse_args()
    #
    # evaluate_explicit_detector(VanillaJailbreakDetector(), log_file=args.log_file)

    evaluate_defense_with_response(task_agency=CoTV3,
                                   defense_agency=ExplicitMultiAgentDefense,
                                   defense_output_name="/tmp/tmp.json",
                                   model_name="gpt-3.5-turbo-1106")
