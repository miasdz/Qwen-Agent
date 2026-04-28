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
from typing import Union

from .azure import TextChatAtAzure
from .base import LLM_REGISTRY, BaseChatModel, ModelServiceError
from .oai import TextChatAtOAI
from .openvino import OpenVINO
from .qwen_dashscope import QwenChatAtDS
from .qwenaudio_dashscope import QwenAudioChatAtDS
from .qwenomni_oai import QwenOmniChatAtOAI
from .qwenvl_dashscope import QwenVLChatAtDS
from .qwenvl_oai import QwenVLChatAtOAI
from .qwenvlo_dashscope import QwenVLoChatAtDS
from .transformers_llm import Transformers


def get_chat_model(cfg: Union[dict, str] = 'qwen-plus') -> BaseChatModel:
    """The interface of instantiating LLM objects.

    Args:
        cfg: The LLM configuration, one example is:
          cfg = {
              # Use the model service provided by DashScope:
              'model': 'qwen-max',
              'model_server': 'dashscope',

              # Use your own model service compatible with OpenAI API:
              # 'model': 'Qwen',
              # 'model_server': 'http://127.0.0.1:7905/v1',

              # (Optional) LLM hyper-parameters:
              'generate_cfg': {
                  'top_p': 0.8,
                  'max_input_tokens': 6500,
                  'max_retries': 10,
              }
          }

    Returns:
        LLM object.
    """
    # 如果传入的是字符串（模型名称），转换为字典配置
    if isinstance(cfg, str):
        cfg = {'model': cfg}

    # 情况1：如果配置中明确指定了 model_type，直接使用注册的对应类创建实例
    if 'model_type' in cfg:
        model_type = cfg['model_type']
        if model_type in LLM_REGISTRY:
            # 特殊处理：对于 OpenAI 兼容接口，如果指定使用 dashscope，则转换为实际的 API 地址
            if model_type in ('oai', 'qwenvl_oai'):
                if cfg.get('model_server', '').strip() == 'dashscope':
                    cfg = copy.deepcopy(cfg)
                    cfg['model_server'] = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
            # 从注册表中获取对应的 LLM 类并实例化
            return LLM_REGISTRY[model_type](cfg)
        else:
            raise ValueError(f'Please set model_type from {str(LLM_REGISTRY.keys())}')

    # 情况2：如果未提供 model_type，根据 model 和 model_server 自动推断

    # 推断规则1：检查是否为 Azure OpenAI
    if 'azure_endpoint' in cfg:
        model_type = 'azure'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    # 推断规则2：如果提供了 model_server 且以 http 开头，视为 OpenAI 兼容接口
    if 'model_server' in cfg:
        if cfg['model_server'].strip().startswith('http'):
            model_type = 'oai'
            cfg['model_type'] = model_type
            return LLM_REGISTRY[model_type](cfg)

    # 获取模型名称用于后续判断
    model = cfg.get('model', '')

    # 推断规则3：根据模型名称中的特征字段判断模型类型
    # 视觉语言模型（如 qwen-vl）
    if '-vl' in model.lower():
        model_type = 'qwenvl_dashscope'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    # 音频模型（如 qwen-audio）
    if '-audio' in model.lower():
        model_type = 'qwenaudio_dashscope'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    # Qwen 系列文本模型（如 qwen-plus、qwen-max）
    if 'qwen' in model.lower():
        model_type = 'qwen_dashscope'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    # 如果以上规则都不匹配，抛出错误
    raise ValueError(f'Invalid model cfg: {cfg}')



__all__ = [
    'BaseChatModel',
    'QwenChatAtDS',
    'TextChatAtOAI',
    'TextChatAtAzure',
    'QwenVLChatAtDS',
    'QwenVLChatAtOAI',
    'QwenAudioChatAtDS',
    'QwenVLoChatAtDS',
    'QwenOmniChatAtOAI',
    'OpenVINO',
    'Transformers',
    'get_chat_model',
    'ModelServiceError',
]
