import re
from typing import Optional, List, Dict, Tuple, Union

import autogen
import openai
from autogen import Agent, UserProxyAgent, OpenAIWrapper

from defense.explicit_detector.explicit_defense_arch import ExplicitMultiAgentDefense, DefenseAssistantAgent, \
    DefenseGroupChat
from defense.utility import load_defense_prompt
from evaluator.evaluate_helper import evaluate_defense_with_response


class CoordinatorAgent(DefenseAssistantAgent):
    def __init__(self, **kwargs):
        kwargs["name"] = "Coordinator"
        self.defense_strategy_name = kwargs.pop("defense_strategy_name")
        super().__init__(**kwargs)
        self.defense_prompt = load_defense_prompt()
        self.register_reply([Agent, None], CoordinatorAgent.generate_coordinate_reply)

    def generate_coordinate_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        last_agent_name = self._oai_messages[sender][-1]['name']
        if last_agent_name == "TaskInputAgent":
            final = True
            response = self.defense_prompt[self.defense_strategy_name]["1_user"]
        elif last_agent_name == "IntentionAnalyzer":
            final = True
            system_input = re.findall(r"--SYSTEM INPUT START--\n((.|\n)*)--SYSTEM INPUT END--",
                                      self._oai_messages[sender][0]['content'])[0][0]
            response = self.defense_prompt[self.defense_strategy_name]["2_user"].replace("[INSERT INPUT HERE]", system_input)
        elif last_agent_name == "Judge":
            final = True
            response = "Next: TaskInputAgent"
        else:
            raise ValueError("Unknown agent name", last_agent_name)
        return final, response


class AutoGenDetectorV1:
    def __init__(self, config_list, defense_strategy_name="explicit_2_agent"):
        self.defense_prompt = load_defense_prompt()
        config = {
            "cache_seed": 123,  # change the cache_seed for different trials
            "config_list": config_list,
            "timeout": 300,
        }
        self.user_input_agent = UserProxyAgent(name="TaskInputAgent",
                                               human_input_mode="NEVER",
                                               is_termination_msg=lambda x: True)
        self.coordinator = CoordinatorAgent(
            llm_config=config,
            defense_strategy_name=defense_strategy_name
        )
        self.analyzer = DefenseAssistantAgent(
            name="IntentionAnalyzer",
            system_message=self.defense_prompt[defense_strategy_name]["1_system"],
            llm_config=config
        )
        self.judge = DefenseAssistantAgent(
            name="Judge",
            system_message=self.defense_prompt[defense_strategy_name]["2_system"],
            llm_config=config
        )
        groupchat = DefenseGroupChat(
            agents=[self.user_input_agent, self.coordinator, self.analyzer, self.judge],
            messages=[], max_round=10)
        self.manager = autogen.GroupChatManager(groupchat=groupchat,
                                                llm_config=config)
        self.defense_strategy_name = defense_strategy_name

    def clear_history(self):
        self.user_input_agent.clear_history()
        self.analyzer.clear_history()
        self.judge.clear_history()
        self.coordinator.clear_history()
        self.manager.clear_history()

    def __call__(self, content_text, user_prompt=None):
        self.clear_history()
        try:
            self.user_input_agent.initiate_chat(self.manager,
                                                message=self.defense_prompt[self.defense_strategy_name]["0_user"].
                                                replace("[INSERT INPUT HERE]", content_text),
                                                clear_history=True, silent=True)
            llm_judgement_output = self.manager._oai_messages[self.judge][-2]['content']
            success = True
            if "INVALID" in llm_judgement_output:
                valid = False
            elif "VALID" in llm_judgement_output:
                valid = True
            else:
                valid = False
                success = False
        except openai.BadRequestError as e:
            valid = False
            success = True
            llm_judgement_output = "ERROR" + str(e)

        return valid, success, llm_judgement_output

    def agency_execute(self, llm_output):
        valid, success, llm_output = self(llm_output)
        return llm_output


class AutoGenDetectorV0125(AutoGenDetectorV1):
    def __init__(self, config_list):
        super().__init__(config_list, defense_strategy_name="explicit_2_agent_v0125")


if __name__ == "__main__":
    # args = argparse.ArgumentParser()
    # args.add_argument("--log_file", type=str, default="results/detection_summary.json")
    # args.add_argument("--port", type=int, default=8005)
    # args = args.parse_args()
    #
    # detector = AutoGenDetectorV1(port=args.port)
    # evaluate_explicit_detector(detector, log_file=args.log_file)

    evaluate_defense_with_response(task_agency=AutoGenDetectorV1,
                                   defense_agency=ExplicitMultiAgentDefense,
                                   defense_output_name="data/defense_output/openai/gpt-3.5-turbo-1106/ex-2.json",
                                   model_name="gpt-3.5-turbo-1106")
