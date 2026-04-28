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

import ast
import os
from typing import List, Literal

# LLM 相关配置
DEFAULT_MAX_INPUT_TOKENS: int = int(os.getenv(
    'QWEN_AGENT_DEFAULT_MAX_INPUT_TOKENS', 58000))

# Agent 执行控制配置
MAX_LLM_CALL_PER_RUN: int = int(os.getenv('QWEN_AGENT_MAX_LLM_CALL_PER_RUN', 20))

# 工具相关工作目录配置
DEFAULT_WORKSPACE: str = os.getenv('QWEN_AGENT_DEFAULT_WORKSPACE', 'workspace')

# RAG（检索增强生成）相关配置
DEFAULT_MAX_REF_TOKEN: int = int(os.getenv('QWEN_AGENT_DEFAULT_MAX_REF_TOKEN',
                                           20000))
DEFAULT_PARSER_PAGE_SIZE: int = int(os.getenv('QWEN_AGENT_DEFAULT_PARSER_PAGE_SIZE',
                                              500))
DEFAULT_RAG_KEYGEN_STRATEGY: Literal['None', 'GenKeyword', 'SplitQueryThenGenKeyword', 'GenKeywordWithKnowledge',
                                     'SplitQueryThenGenKeywordWithKnowledge'] = os.getenv(
                                         'QWEN_AGENT_DEFAULT_RAG_KEYGEN_STRATEGY', 'GenKeyword')
DEFAULT_RAG_SEARCHERS: List[str] = ast.literal_eval(
    os.getenv('QWEN_AGENT_DEFAULT_RAG_SEARCHERS',
              "['keyword_search', 'front_page_search']"))
