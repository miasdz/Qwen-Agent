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

from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, field_validator, model_validator

DEFAULT_SYSTEM_MESSAGE = ''

ROLE = 'role'
CONTENT = 'content'
REASONING_CONTENT = 'reasoning_content'
NAME = 'name'

SYSTEM = 'system'
USER = 'user'
ASSISTANT = 'assistant'
FUNCTION = 'function'

FILE = 'file'
IMAGE = 'image'
AUDIO = 'audio'
VIDEO = 'video'


class BaseModelCompatibleDict(BaseModel):

    @log_execution
    def __getitem__(self, item):
        return getattr(self, item)

    @log_execution
    def __setitem__(self, key, value):
        setattr(self, key, value)

    @log_execution
    def model_dump(self, **kwargs):
        if 'exclude_none' not in kwargs:
            kwargs['exclude_none'] = True
        return super().model_dump(**kwargs)

    @log_execution
    def model_dump_json(self, **kwargs):
        if 'exclude_none' not in kwargs:
            kwargs['exclude_none'] = True
        return super().model_dump_json(**kwargs)

    @log_execution
    def get(self, key, default=None):
        try:
            value = getattr(self, key)
            if value:
                return value
            else:
                return default
        except AttributeError:
            return default

    @log_execution
    def __str__(self):
        return f'{self.model_dump()}'


class FunctionCall(BaseModelCompatibleDict):
    name: str
    arguments: str

    @log_execution
    def __init__(self, name: str, arguments: str):
        super().__init__(name=name, arguments=arguments)

    @log_execution
    def __repr__(self):
        return f'FunctionCall({self.model_dump()})'


class ContentItem(BaseModelCompatibleDict):
    text: Optional[str] = None
    image: Optional[str] = None
    file: Optional[str] = None
    audio: Optional[Union[str, dict]] = None
    video: Optional[Union[str, list]] = None

    @log_execution
    def __init__(self,
                 text: Optional[str] = None,
                 image: Optional[str] = None,
                 file: Optional[str] = None,
                 audio: Optional[Union[str, dict]] = None,
                 video: Optional[Union[str, list]] = None):
        super().__init__(text=text, image=image, file=file, audio=audio, video=video)

    @model_validator(mode='after')
    @log_execution
    def check_exclusivity(self):
        provided_fields = 0
        if self.text is not None:
            provided_fields += 1
        if self.image:
            provided_fields += 1
        if self.file:
            provided_fields += 1
        if self.audio:
            provided_fields += 1
        if self.video:
            provided_fields += 1

        if provided_fields != 1:
            raise ValueError("Exactly one of 'text', 'image', 'file', 'audio', or 'video' must be provided.")
        return self

    @log_execution
    def __repr__(self):
        return f'ContentItem({self.model_dump()})'

    @log_execution
    def get_type_and_value(self) -> Tuple[Literal['text', 'image', 'file', 'audio', 'video'], str]:
        (t, v), = self.model_dump().items()
        assert t in ('text', 'image', 'file', 'audio', 'video')
        return t, v

    @property
    @log_execution
    def type(self) -> Literal['text', 'image', 'file', 'audio', 'video']:
        t, v = self.get_type_and_value()
        return t

    @property
    @log_execution
    def value(self) -> str:
        t, v = self.get_type_and_value()
        return v


class Message(BaseModelCompatibleDict):
    role: str
    content: Union[str, List[ContentItem]]
    reasoning_content: Optional[Union[str, List[ContentItem]]] = None
    name: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    extra: Optional[dict] = None

    @log_execution
    def __init__(self,
                 role: str,
                 content: Union[str, List[ContentItem]],
                 reasoning_content: Optional[Union[str, List[ContentItem]]] = None,
                 name: Optional[str] = None,
                 function_call: Optional[FunctionCall] = None,
                 extra: Optional[dict] = None,
                 **kwargs):
        if content is None:
            content = ''
        super().__init__(role=role,
                         content=content,
                         reasoning_content=reasoning_content,
                         name=name,
                         function_call=function_call,
                         extra=extra)

    @log_execution
    def __repr__(self):
        return f'Message({self.model_dump()})'

    @field_validator('role')
    @log_execution
    def role_checker(cls, value: str) -> str:
        if value not in [USER, ASSISTANT, SYSTEM, FUNCTION]:
            raise ValueError(f'{value} must be one of {",".join([USER, ASSISTANT, SYSTEM, FUNCTION])}')
        return value
