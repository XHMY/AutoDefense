import re
from typing import *

import autogen
import openai
from autogen import Agent, ConversableAgent, UserProxyAgent, AssistantAgent, OpenAIWrapper

from defense.utility import load_defense_prompt


class DefenseGroupChat(autogen.GroupChat):
    def select_speaker(self, last_speaker: Agent, selector: ConversableAgent):
        if last_speaker.name != "Coordinator":
            return self.agent_by_name("Coordinator")
        else:
            mentions = self._mentioned_agents(self.agent_by_name("Coordinator").last_message()['content'], self.agents)
            assert len(mentions) == 1, "Coordinator should only mention one agent"
            name = next(iter(mentions))
            return self.agent_by_name(name)


class DefenseAssistantAgent(AssistantAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_reply([Agent, None], DefenseAssistantAgent.generate_oai_reply)

    def generate_oai_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        final, response = super().generate_oai_reply(messages, sender, config)
        if response is None:
            return final, "Due to Azure OpenAI's content management policy, the response is too sensitive to be shown."
        return final, response


class TaskAgencyAgent(ConversableAgent):
    def __init__(self, agency, **kwargs):
        super().__init__(**kwargs)
        self.register_reply([Agent, None], TaskAgencyAgent.execute_agency)
        self.agency = agency

    def execute_agency(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, None]]:
        client = self.client if config is None else config
        if client is None:
            raise ValueError("client is None")
        if messages is None:
            raise ValueError("messages is None")

        response = self.agency.agency_execute(messages[-1]["content"])

        return True, response


class OutputAgent(ConversableAgent):
    def __init__(self, strict_sorry_for_invalid, **kwargs):
        super().__init__(**kwargs)
        self.strict_sorry_for_invalid = strict_sorry_for_invalid
        self.register_reply([Agent, None], OutputAgent.refine_invalid_output)
        self.register_reply([Agent, None], OutputAgent.exam_valid_output)

    def exam_valid_output(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, None]]:
        if messages is None:
            raise ValueError("messages is None")

        if "I apologize" in messages[-1]["content"]:
            return True, "I'm sorry, but I'm not able to provide the information you're looking for."

        valid_in = "VALID" in messages[-1]["content"].upper()
        invalid_in = "INVALID" in messages[-1]["content"].upper()

        if valid_in and not invalid_in:
            if len([m for m in messages if m["name"] == 'LLM_Victim_Agent']) != 0:
                return True, [m for m in messages if m["name"] == 'LLM_Victim_Agent'][0]["content"]
            else:
                first_input = [m for m in messages if m["name"] == 'User_Prompt_Agent'][0]["content"]
                first_input = re.search(r'--SYSTEM INPUT START--\n((.|\n)*)\n--SYSTEM INPUT END--', first_input).group(
                    1)
                return True, first_input
        elif invalid_in:
            if self.strict_sorry_for_invalid:
                return True, "I'm sorry, but I'm not able to provide the information you're looking for."
            else:
                return False, "The output needs refinement."
        else:
            if "Azure OpenAI's content management policy" in messages[-1]["content"]:
                return True, "I'm sorry, but I'm not able to provide the information you're looking for."
            else:
                return True, f"DEFENSE ERROR: Judgement not exist. Output: {messages[-1]['content']}"

    def refine_invalid_output(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, None]]:
        client = self.client if config is None else config
        if client is None:
            raise ValueError("client is None")
        if messages is None:
            raise ValueError("messages is None")

        final, response = self.generate_oai_reply(messages, sender, config)
        return final, response


class LLMVitimAgent(ConversableAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_reply([Agent, None], LLMVitimAgent.generate_oai_azure_reply)

    def generate_oai_azure_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, None]]:

        try:
            response = super().generate_oai_reply(messages, sender, config)
        except openai.BadRequestError as e:
            if e.status_code == 400:
                return True, "I'm sorry, but I'm not able to provide the information you're looking for. (From Azure Content Management)"
        return response


class ExplicitMultiAgentDefense:
    def __init__(self, task_agency, config_list, cache_seed=123):
        self.config_list = config_list
        self.cache_seed = cache_seed
        self.defense_prompt = load_defense_prompt()
        self.user_input_agent = UserProxyAgent(name="User_Prompt_Agent", is_termination_msg=lambda x: True,
                                               human_input_mode="NEVER",
                                               llm_config={"config_list": config_list, "cache_seed": cache_seed})
        self.llm_agent = LLMVitimAgent(name="LLM_Victim_Agent",
                                       system_message="You are a helpful AI assistant.",
                                       llm_config={"config_list": config_list, "cache_seed": cache_seed})

        self.taskagency_agent = TaskAgencyAgent(task_agency, name="Task_Agency_Agent",
                                                llm_config={"config_list": config_list, "cache_seed": cache_seed})
        self.output_agent = OutputAgent(True, name="Output_Agent",
                                        system_message="Your task is to refuse to answer user's question. Do not mention the invalid content in your response.",
                                        llm_config={"config_list": config_list, "cache_seed": cache_seed})

    def clear_history(self):
        self.user_input_agent.clear_history()
        self.llm_agent.clear_history()
        self.taskagency_agent.clear_history()
        self.output_agent.clear_history()

    def defense_with_response(self, response):
        self.clear_history()
        groupchat = autogen.GroupChat(
            agents=[self.user_input_agent, self.taskagency_agent, self.output_agent], messages=[],
            max_round=4, speaker_selection_method="round_robin")
        manager = autogen.GroupChatManager(groupchat=groupchat,
                                           llm_config={"config_list": self.config_list, "cache_seed": self.cache_seed})
        self.user_input_agent.initiate_chat(manager, message=self.defense_prompt["explicit_defense"]["1_user"].
                                            replace("[INSERT INPUT HERE]", response), silent=True)
        result = self.output_agent.last_message()

        find_reg_output = re.search(r'--SYSTEM OUTPUT START--\n((.|\n)*)\n--SYSTEM OUTPUT END--', result["content"])
        if find_reg_output is not None:
            result["content"] = find_reg_output.group(1)

        return result

    def defense_with_prompt(self, prompt):
        self.clear_history()
        groupchat = autogen.GroupChat(
            agents=[self.user_input_agent, self.llm_agent, self.taskagency_agent, self.output_agent], messages=[],
            max_round=4, speaker_selection_method="round_robin")
        manager = autogen.GroupChatManager(groupchat=groupchat,
                                           llm_config={"config_list": self.config_list, "cache_seed": self.cache_seed})
        self.user_input_agent.initiate_chat(manager, message=prompt)
        result = self.output_agent.last_message()
        return result
