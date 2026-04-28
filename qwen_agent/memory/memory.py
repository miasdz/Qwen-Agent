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

import json
from importlib import import_module
from typing import Dict, Iterator, List, Optional, Union

import json5

from qwen_agent import Agent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, USER, Message
from qwen_agent.log import logger
from qwen_agent.settings import (DEFAULT_MAX_REF_TOKEN, DEFAULT_PARSER_PAGE_SIZE, DEFAULT_RAG_KEYGEN_STRATEGY,
                                 DEFAULT_RAG_SEARCHERS)
from qwen_agent.tools import BaseTool
from qwen_agent.tools.simple_doc_parser import PARSER_SUPPORTED_FILE_TYPES
from qwen_agent.utils.utils import extract_files_from_messages, extract_text_from_message, get_file_type


class Memory(Agent):
    """Memory is special agent for file management.

    By default, this memory can use retrieval tool for RAG.
    """

    @log_execution
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 files: Optional[List[str]] = None,
                 rag_cfg: Optional[Dict] = None):
        """初始化记忆模块，用于文件管理和 RAG（检索增强生成）功能。

        该构造函数配置 RAG 相关参数，并初始化工具列表，包括文档检索和文档解析工具。
        如果没有提供 LLM，则禁用关键词生成策略。

        Args:
            function_list: 可选的工具列表，支持工具名称、工具配置字典或 BaseTool 实例。
            llm: 大语言模型配置或实例，用于关键词生成等任务。
            system_message: 系统消息，默认为 DEFAULT_SYSTEM_MESSAGE。
            files: 可选的文件列表，这些文件将被添加到系统中进行管理。
            rag_cfg: RAG 配置字典，支持以下配置项：
                - 'max_ref_token': 最大参考 token 数量，默认为 DEFAULT_MAX_REF_TOKEN
                - 'parser_page_size': 解析器页面大小，默认为 DEFAULT_PARSER_PAGE_SIZE
                - 'rag_keygen_strategy': RAG 关键词生成策略，默认为 DEFAULT_RAG_KEYGEN_STRATEGY
                - 'rag_searchers': RAG 搜索器列表，默认为 DEFAULT_RAG_SEARCHERS

                示例配置：
                {
                    'max_ref_token': 4000,
                    'parser_page_size': 500,
                    'rag_keygen_strategy': 'SplitQueryThenGenKeyword',
                    'rag_searchers': ['keyword_search', 'front_page_search']
                }

        Raises:
            无显式异常抛出，但父类 Agent 的初始化可能会抛出异常。
        """
        # 初始化 RAG 配置参数
        self.cfg = rag_cfg or {}
        self.max_ref_token: int = self.cfg.get('max_ref_token', DEFAULT_MAX_REF_TOKEN)
        self.parser_page_size: int = self.cfg.get('parser_page_size', DEFAULT_PARSER_PAGE_SIZE)
        self.rag_searchers = self.cfg.get('rag_searchers', DEFAULT_RAG_SEARCHERS)
        self.rag_keygen_strategy = self.cfg.get('rag_keygen_strategy', DEFAULT_RAG_KEYGEN_STRATEGY)

        # 如果没有可用的 LLM，则禁用关键词生成策略
        if not llm:
            # There is no suitable model available for keygen
            self.rag_keygen_strategy = 'none'

        # 初始化工具列表，添加默认的检索和文档解析工具
        function_list = function_list or []
        super().__init__(function_list=[{
            'name': 'retrieval',
            'max_ref_token': self.max_ref_token,
            'parser_page_size': self.parser_page_size,
            'rag_searchers': self.rag_searchers,
        }, {
            'name': 'doc_parser',
            'max_ref_token': self.max_ref_token,
            'parser_page_size': self.parser_page_size,
        }] + function_list,
                         llm=llm,
                         system_message=system_message)

        self.system_files = files or []

    @log_execution
    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """This agent is responsible for processing the input files in the message.

         This method stores the files in the knowledge base, and retrievals the relevant parts
         based on the query and returning them.
         The currently supported file types include: .pdf, .docx, .pptx, .txt, .csv, .tsv, .xlsx, .xls and html.

         Args:
             messages: A list of messages.
             lang: Language.

        Yields:
            The message of retrieved documents.
        """
        # process files in messages
        rag_files = self.get_rag_files(messages)

        if not rag_files:
            yield [Message(role=ASSISTANT, content='', name='memory')]
        else:
            query = ''
            # Only retrieval content according to the last user query if exists
            if messages and messages[-1].role == USER:
                query = extract_text_from_message(messages[-1], add_upload_info=False)

            # Keyword generation
            if query and self.rag_keygen_strategy.lower() != 'none':
                module_name = 'qwen_agent.agents.keygen_strategies'
                module = import_module(module_name)
                cls = getattr(module, self.rag_keygen_strategy)
                keygen = cls(llm=self.llm)
                response = keygen.run([Message(USER, query)], files=rag_files)
                last = None
                for last in response:
                    continue
                if last:
                    keyword = last[-1].content.strip()
                else:
                    keyword = ''

                if keyword.startswith('```json'):
                    keyword = keyword[len('```json'):]
                if keyword.endswith('```'):
                    keyword = keyword[:-3]
                try:
                    keyword_dict = json5.loads(keyword)
                    if 'text' not in keyword_dict:
                        keyword_dict['text'] = query
                    query = json.dumps(keyword_dict, ensure_ascii=False)
                    logger.info(query)
                except Exception:
                    query = query

            content = self.function_map['retrieval'].call(
                {
                    'query': query,
                    'files': rag_files
                },
                **kwargs,
            )
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, indent=4)

            yield [Message(role=ASSISTANT, content=content, name='memory')]

    @log_execution
    def get_rag_files(self, messages: List[Message]):
        session_files = extract_files_from_messages(messages, include_images=False)
        files = self.system_files + session_files
        rag_files = []
        for file in files:
            f_type = get_file_type(file)
            if f_type in PARSER_SUPPORTED_FILE_TYPES and file not in rag_files:
                rag_files.append(file)
        return rag_files
