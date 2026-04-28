from qwen_agent.log_util import log_execution
# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple, Union

from qwen_agent.llm import get_chat_model
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, ROLE, SYSTEM, ContentItem, Message
from qwen_agent.log import logger
from qwen_agent.tools import TOOL_REGISTRY, BaseTool, MCPManager
from qwen_agent.tools.base import ToolServiceError
from qwen_agent.tools.simple_doc_parser import DocParserError
from qwen_agent.utils.utils import has_chinese_messages, merge_generate_cfgs


class Agent(ABC):
    """A base class for Agent.

    An agent can receive messages and provide response by LLM or Tools.
    Different agents have distinct workflows for processing messages and generating responses in the `_run` method.
    """

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 **kwargs):
        """Initialization the agent.
        ** kwargs 会将所有未被函数定义捕获的关键字参数（即key = value形式的参数）收集起来，并打包成一个字典(dictionary)。

        Args:
            function_list: One list of tool name, tool configuration or Tool object,
              such as 'code_interpreter', {'name': 'code_interpreter', 'timeout': 10}, or CodeInterpreter().
            llm: The LLM model configuration or LLM model object.
              Set the configuration as {'model': '', 'api_key': '', 'model_server': ''}.
            system_message: The specified system message for LLM chat.
            name: The name of this agent.
            description: The description of this agent, which will be used for multi_agent.
        """
        if isinstance(llm, dict):
            self.llm = get_chat_model(llm)
        else:
            self.llm = llm
        self.extra_generate_cfg: dict = {}

        self.function_map = {}
        if function_list:
            for tool in function_list:
                self._init_tool(tool)

        self.system_message = system_message
        self.name = name
        self.description = description

    @log_execution
    def run_nonstream(self, messages: List[Union[Dict, Message]], **kwargs) -> Union[List[Message], List[Dict]]:
        """Same as self.run, but with stream=False,
        meaning it returns the complete response directly
        instead of streaming the response incrementally."""
        *_, last_responses = self.run(messages, **kwargs)
        return last_responses

    @log_execution
    def run(self, messages: List[Union[Dict, Message]],
            **kwargs) -> Union[Iterator[List[Message]], Iterator[List[Dict]]]:
        """Return one response generator based on the received messages.

        This method performs a uniform type conversion for the inputted messages,
        and calls the _run method to generate a reply.

        Args:
            messages: A list of messages.

        Yields:
            The response generator.
        """
        # 深拷贝输入消息列表，避免修改原始数据
        messages = copy.deepcopy(messages)
        # 设置默认返回类型为字典格式
        _return_message_type = 'dict'
        # 初始化新的消息列表，用于存储类型统一后的消息
        new_messages = []
        # 如果输入消息列表为空，默认返回 Message 对象类型
        if not messages:
            _return_message_type = 'message'
        # 遍历所有输入消息，进行类型统一转换
        for msg in messages:
            # 如果当前消息是字典类型，将其转换为 Message 对象
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            # 如果当前消息已经是 Message 对象，直接使用
            else:
                new_messages.append(msg)
                # 只要有一个消息是 Message 对象，就将返回类型设置为 message
                _return_message_type = 'message'

        # 如果调用者没有指定语言参数，根据消息内容自动检测
        if 'lang' not in kwargs:
            # 检测消息中是否包含中文，自动设置合适的语言
            if has_chinese_messages(new_messages):
                kwargs['lang'] = 'zh'
            else:
                kwargs['lang'] = 'en'

        # 如果代理定义了系统指令消息（system_message）
        if self.system_message:
            # 检查消息列表是否为空或第一条消息不是系统消息
            if not new_messages or new_messages[0][ROLE] != SYSTEM:
                # 在消息列表开头插入系统指令
                new_messages.insert(0, Message(role=SYSTEM, content=self.system_message))
            # 如果第一条消息已经是系统消息，需要合并系统指令
            else:
                # 如果现有系统消息是字符串类型，直接拼接
                if isinstance(new_messages[0][CONTENT], str):
                    new_messages[0][CONTENT] = self.system_message + '\n\n' + new_messages[0][CONTENT]
                # 如果现有系统消息是列表类型（多模态内容）
                else:
                    # 断言：确保内容是 ContentItem 列表且第一个元素有文本
                    assert isinstance(new_messages[0][CONTENT], list)
                    assert new_messages[0][CONTENT][0].text
                    # 在列表开头插入新的系统指令文本项
                    new_messages[0][CONTENT] = [ContentItem(text=self.system_message + '\n\n')
                                               ] + new_messages[0][CONTENT]  # noqa

        # 调用子类实现的 _run 方法，流式获取代理的响应
        for rsp in self._run(messages=new_messages, **kwargs):
            # 遍历响应中的每条消息
            for i in range(len(rsp)):
                # 如果消息没有名称且代理有定义名称，自动填充代理名称
                if not rsp[i].name and self.name:
                    rsp[i].name = self.name
            # 根据返回类型决定输出格式
            if _return_message_type == 'message':
                # 如果输入包含 Message 对象，返回 Message 对象列表
                yield [Message(**x) if isinstance(x, dict) else x for x in rsp]
            else:
                # 如果输入全是字典，返回字典列表（将 Message 对象转换为字典）
                yield [x.model_dump() if not isinstance(x, dict) else x for x in rsp]


    @abstractmethod
    @log_execution
    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """Return one response generator based on the received messages.

        The workflow for an agent to generate a reply.
        Each agent subclass needs to implement this method.

        Args:
            messages: A list of messages.
            lang: Language, which will be used to select the language of the prompt
              during the agent's execution process.

        Yields:
            The response generator.
        """
        raise NotImplementedError

    @log_execution
    def _call_llm(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        stream: bool = True,
        extra_generate_cfg: Optional[dict] = None,
    ) -> Iterator[List[Message]]:
        """The interface of calling LLM for the agent.

        We prepend the system_message of this agent to the messages, and call LLM.

        Args:
            messages: A list of messages.
            functions: The list of functions provided to LLM.
            stream: LLM streaming output or non-streaming output.
              For consistency, we default to using streaming output across all agents.

        Yields:
            The response generator of LLM.
        """
        return self.llm.chat(messages=messages,
                             functions=functions,
                             stream=stream,
                             extra_generate_cfg=merge_generate_cfgs(
                                 base_generate_cfg=self.extra_generate_cfg,
                                 new_generate_cfg=extra_generate_cfg,
                             ))

    @log_execution
    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> Union[str, List[ContentItem]]:
        """The interface of calling tools for the agent.

        Args:
            tool_name: The name of one tool.
            tool_args: Model generated or user given tool parameters.

        Returns:
            The output of tools.
        """
        if tool_name not in self.function_map:
            return f'Tool {tool_name} does not exists.'
        tool = self.function_map[tool_name]
        try:
            tool_result = tool.call(tool_args, **kwargs)
        except (ToolServiceError, DocParserError) as ex:
            raise ex
        except Exception as ex:
            exception_type = type(ex).__name__
            exception_message = str(ex)
            traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
            error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}\n' \
                            f'Traceback:\n{traceback_info}'
            logger.warning(error_message)
            return error_message

        if isinstance(tool_result, str):
            return tool_result
        elif isinstance(tool_result, list) and all(isinstance(item, ContentItem) for item in tool_result):
            return tool_result  # multimodal tool results
        else:
            return json.dumps(tool_result, ensure_ascii=False, indent=4)

    @log_execution
    def _init_tool(self, tool: Union[str, Dict, BaseTool]):
        """初始化工具并注册到代理的函数映射表中。

        该方法支持三种类型的工具输入：
        1. BaseTool 实例：直接注册到函数映射表
        2. 包含 MCP 服务器配置的字典：通过 MCPManager 初始化多个工具
        3. 工具名称字符串或普通工具配置字典：从 TOOL_REGISTRY 中查找并实例化工具

        如果工具已存在，会发出警告并覆盖现有工具。

        Args:
            tool: 要初始化的工具，可以是以下类型之一：
                - str: 工具名称，将从 TOOL_REGISTRY 中查找并实例化
                - Dict: 工具配置字典，可能包含：
                    * 'mcpServers' 键：表示 MCP 服务器配置
                    * 'name' 键及其他配置项：普通工具配置
                - BaseTool: BaseTool 的实例对象

        Raises:
            ValueError: 当提供的工具名称未在 TOOL_REGISTRY 中注册时抛出
        """
        # 处理 BaseTool 实例的情况
        if isinstance(tool, BaseTool):
            tool_name = tool.name
            if tool_name in self.function_map:
                logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
            self.function_map[tool_name] = tool
        # 处理包含 MCP 服务器配置的字典
        elif isinstance(tool, dict) and 'mcpServers' in tool:
            tools = MCPManager().initConfig(tool)
            for tool in tools:
                tool_name = tool.name
                if tool_name in self.function_map:
                    logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
                self.function_map[tool_name] = tool
        # 处理普通工具配置或工具名称
        else:
            if isinstance(tool, dict):
                tool_name = tool['name']
                tool_cfg = tool
            else:
                tool_name = tool
                tool_cfg = None
            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f'Tool {tool_name} is not registered.')

            if tool_name in self.function_map:
                logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
            self.function_map[tool_name] = TOOL_REGISTRY[tool_name](tool_cfg)

    @log_execution
    def _detect_tool(self, message: Message) -> Tuple[bool, str, str, str]:
        """A built-in tool call detection for func_call format message.

        Args:
            message: one message generated by LLM.

        Returns:
            Need to call tool or not, tool name, tool args, text replies.
        """
        func_name = None
        func_args = None

        if message.function_call:
            func_call = message.function_call
            func_name = func_call.name
            func_args = func_call.arguments
        text = message.content
        if not text:
            text = ''

        return (func_name is not None), func_name, func_args, text


# The most basic form of an agent is just a LLM, not augmented with any tool or workflow.
class BasicAgent(Agent):

    @log_execution
    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        extra_generate_cfg = {'lang': lang}
        if kwargs.get('seed') is not None:
            extra_generate_cfg['seed'] = kwargs['seed']
        return self._call_llm(messages, extra_generate_cfg=extra_generate_cfg)
