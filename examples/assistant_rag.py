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
from pathlib import Path

from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI


def test():
    bot = Assistant(llm={'model': 'qwen-plus-latest'})
    messages = [{'role': 'user', 'content': [{'text': '介绍图一'}, {'file': 'https://arxiv.org/pdf/1706.03762.pdf'}]}]
    for rsp in bot.run(messages):
        print(rsp)


def app_gui():
    llm_cfg = {
        'model': 'qwen3-max-2026-01-23',
        'model_type': 'qwen_dashscope',
        'api_key': 'sk-a4cf2f63ea094d419d01829800fc68bf',
        'generate_cfg': {
            'top_p': 0.8
        }
    }

    TARGET_DIR = Path(__file__).parent
    files = list(TARGET_DIR.glob("data/*"))

    files_list = [file.as_posix().strip() for file in files]

    # Define the agent
    bot = Assistant(llm=llm_cfg,
                    name='Assistant',
                    description='使用RAG检索并回答，支持文件类型：PDF/Word/PPT/TXT/HTML。',
                    files=files_list)
    chatbot_config = {
        'prompt.suggestions': [
            {
                'text': '介绍图一'
            },
            {
                'text': '第二章第一句话是什么？'
            },
        ],
        "verbose": True
    }
    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    # test()
    app_gui()
