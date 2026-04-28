from qwen_agent.log_util import log_execution
from typing import Dict, Optional

from qwen_agent.llm.base import register_llm
from qwen_agent.llm.qwenvl_dashscope import QwenVLChatAtDS


@register_llm('qwenvlo_dashscope')
class QwenVLoChatAtDS(QwenVLChatAtDS):

    @property
    @log_execution
    def support_multimodal_output(self) -> bool:
        return True

    @log_execution
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model = self.model or 'qwen-audio-turbo-latest'
