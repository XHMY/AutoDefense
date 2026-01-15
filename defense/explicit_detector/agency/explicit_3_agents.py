import re
from typing import Optional, List, Dict, Tuple, Union

import autogen
import openai
from autogen import UserProxyAgent, Agent, OpenAIWrapper

from defense.explicit_detector.explicit_defense_arch import ExplicitMultiAgentDefense, DefenseAssistantAgent, \
    DefenseGroupChat
from defense.utility import load_defense_prompt
from evaluator.evaluate_helper import evaluate_defense_with_response


class CoordinatorAgent(DefenseAssistantAgent):
    def __init__(self, **kwargs):
        kwargs["name"] = "Coordinator"
        self.strategy_name = "explicit_3_agent"
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
            response = self.defense_prompt[self.strategy_name]["1_user"]
        elif last_agent_name == "IntentionAnalyzer":
            final = True
            response = self.defense_prompt[self.strategy_name]["2_user"]
        elif last_agent_name == "OriginalPromptAnalyzer":
            final = True
            system_input = re.findall(r"--SYSTEM INPUT START--\n((.|\n)*)--SYSTEM INPUT END--",
                                      self._oai_messages[sender][0]['content'])[0][0]
            response = self.defense_prompt[self.strategy_name]["3_user"].replace("[INSERT INPUT HERE]", system_input)
        elif last_agent_name == "Judge":
            final = True
            response = "Next: TaskInputAgent"
        else:
            raise ValueError("Unknown agent name", last_agent_name)
        return final, response


class CoordinatorAgentV2(DefenseAssistantAgent):
    def __init__(self, **kwargs):
        kwargs["name"] = "Coordinator"
        self.strategy_name = "explicit_3_agent_v2"
        super().__init__(**kwargs)
        self.defense_prompt = load_defense_prompt()
        self.register_reply([Agent, None], CoordinatorAgentV2.generate_coordinate_reply)

    def generate_coordinate_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        last_agent_name = self._oai_messages[sender][-1]['name']
        if last_agent_name == "TaskInputAgent":
            final = True
            response = self.defense_prompt[self.strategy_name]["1_user"]
        elif last_agent_name == "ValidityAssumedIntentionAnalyzer":
            final = True
            response = self.defense_prompt[self.strategy_name]["2_user"]
        elif last_agent_name == "InvalidityAssumedIntentionAnalyzer":
            final = True
            system_input = re.findall(r"--SYSTEM INPUT START--\n((.|\n)*)--SYSTEM INPUT END--",
                                      self._oai_messages[sender][0]['content'])[0][0]
            response = self.defense_prompt[self.strategy_name]["3_user"].replace("[INSERT INPUT HERE]", system_input)
        elif last_agent_name == "Judge":
            final = True
            response = "Next: TaskInputAgent"
        else:
            raise ValueError("Unknown agent name", last_agent_name)
        return final, response


class AutoGenDetectorThreeAgency:
    def __init__(self, config_list):
        self.defense_prompt = load_defense_prompt()
        self.strategy_name = "explicit_3_agent"
        self.config = {
            "cache_seed": 123,  # change the cache_seed for different trials
            "config_list": config_list,
            "timeout": 300,
        }
        self.init_agents()

    def init_agents(self):
        self.user_input_agent = UserProxyAgent(name="TaskInputAgent",
                                               human_input_mode="NEVER",
                                               is_termination_msg=lambda x: True)
        self.coordinator = CoordinatorAgent(
            llm_config=self.config
        )
        self.agent_athena = DefenseAssistantAgent(
            name="IntentionAnalyzer",
            system_message=self.defense_prompt[self.strategy_name]["1_system"],
            llm_config=self.config,
        )
        self.agent_hermes = DefenseAssistantAgent(
            name="OriginalPromptAnalyzer",
            system_message=self.defense_prompt[self.strategy_name]["2_system"],
            llm_config=self.config,
        )
        self.agent_themis = DefenseAssistantAgent(
            name="Judge",
            system_message=self.defense_prompt[self.strategy_name]["3_system"],
            llm_config=self.config
        )
        groupchat = DefenseGroupChat(
            agents=[self.user_input_agent, self.coordinator, self.agent_athena, self.agent_hermes, self.agent_themis],
            messages=[], max_round=20)
        self.manager = autogen.GroupChatManager(groupchat=groupchat,
                                                llm_config=self.config)

    def clear_history(self):
        self.user_input_agent.clear_history()
        self.agent_athena.clear_history()
        self.agent_hermes.clear_history()
        self.coordinator.clear_history()
        self.agent_themis.clear_history()

    def __call__(self, content_text, user_prompt=None):
        self.clear_history()
        try:
            self.user_input_agent.initiate_chat(self.manager, message=self.defense_prompt[self.strategy_name]["0_user"].
                                                replace("[INSERT INPUT HERE]", content_text), silent=False)
            llm_judgement_output = self.manager._oai_messages[self.agent_themis][-2]['content']

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


class AutoGenDetectorThreeAgencyV2(AutoGenDetectorThreeAgency):
    def __init__(self, config_list):
        self.defense_prompt = load_defense_prompt()
        self.strategy_name = "explicit_3_agent_v2"
        self.config = {
            "cache_seed": 123,  # change the cache_seed for different trials
            "config_list": config_list,
            "timeout": 300,
        }
        self.init_agents()

    def init_agents(self):
        self.user_input_agent = UserProxyAgent(name="TaskInputAgent",
                                               human_input_mode="NEVER",
                                               is_termination_msg=lambda x: True)
        self.coordinator = CoordinatorAgentV2(
            llm_config=self.config
        )
        self.agent_athena = DefenseAssistantAgent(
            name="ValidityAssumedIntentionAnalyzer",
            system_message=self.defense_prompt[self.strategy_name]["1_system"],
            llm_config=self.config,
        )
        self.agent_hermes = DefenseAssistantAgent(
            name="InvalidityAssumedIntentionAnalyzer",
            system_message=self.defense_prompt[self.strategy_name]["2_system"],
            llm_config=self.config,
        )
        self.agent_themis = DefenseAssistantAgent(
            name="Judge",
            system_message=self.defense_prompt[self.strategy_name]["3_system"],
            llm_config=self.config
        )
        groupchat = DefenseGroupChat(
            agents=[self.user_input_agent, self.coordinator, self.agent_athena, self.agent_hermes, self.agent_themis],
            messages=[], max_round=20)
        self.manager = autogen.GroupChatManager(groupchat=groupchat,
                                                llm_config=self.config)


if __name__ == "__main__":
    # Example: Evaluate defense with vLLM server
    # Start vLLM server first:
    #   vllm serve meta-llama/Llama-2-13b-chat-hf --port 8000

    evaluate_defense_with_response(task_agency=AutoGenDetectorThreeAgency,
                                   defense_agency=ExplicitMultiAgentDefense,
                                   model_name="llama-2-13b",
                                   chat_file="data/harmful_output/alpaca_data_safe_1000.json",
                                   defense_output_name="/tmp/ex-3.json",
                                   parallel=False,
                                   host_name="127.0.0.1",
                                   port=8000,
                                   frequency_penalty=0.0)
