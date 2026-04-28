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

import os
import pprint
import re
from typing import List, Optional, Union

from qwen_agent import Agent, MultiAgentHub
from qwen_agent.agents.user_agent import PENDING_USER_INPUT
from qwen_agent.gui.gradio_utils import format_cover_html
from qwen_agent.gui.utils import convert_fncall_to_text, convert_history_to_chatbot, get_avatar_image
from qwen_agent.llm.schema import AUDIO, CONTENT, FILE, IMAGE, NAME, ROLE, USER, VIDEO, Message
from qwen_agent.log import logger
from qwen_agent.utils.utils import print_traceback


class WebUI:
    """A Common chatbot application for agent."""

    @log_execution
    def __init__(self, agent: Union[Agent, MultiAgentHub, List[Agent]], chatbot_config: Optional[dict] = None):
        """初始化聊天机器人界面。

        该构造函数配置 WebUI 界面，支持单个或多个代理的展示。
        可以自定义用户和代理的外观、输入提示等界面元素。

        Args:
            agent: 代理对象，支持以下三种类型：
                - Agent: 单个代理实例，如 Assistant、GroupChat、Router 等
                - MultiAgentHub: 多代理中心，可管理多个代理
                - List[Agent]: 代理列表，包含多个代理实例
            chatbot_config: 聊天机器人配置字典，支持以下配置项：
                - 'user.name': 用户名，默认为 'user'
                - 'user.avatar': 用户头像路径，默认根据用户名生成
                - 'agent.avatar': 代理头像路径，默认根据代理名生成
                - 'input.placeholder': 输入框占位符文本，默认为 '跟我聊聊吧～'
                - 'prompt.suggestions': 推荐问题列表，用于展示建议的对话内容
                - 'verbose': 是否显示详细日志，默认为 False

                示例配置：
                {
                    'user.name': '张三',
                    'user.avatar': '/path/to/user_avatar.png',
                    'agent.avatar': '/path/to/agent_avatar.png',
                    'input.placeholder': '请输入您的问题',
                    'prompt.suggestions': ['你好', '帮我分析一下这个文档']
                }

        Raises:
            无显式异常抛出。
        """
        # 处理不同类型的代理输入，统一转换为代理列表
        chatbot_config = chatbot_config or {}

        if isinstance(agent, MultiAgentHub):
            self.agent_list = [agent for agent in agent.nonuser_agents]
            self.agent_hub = agent
        elif isinstance(agent, list):
            self.agent_list = agent
            self.agent_hub = None
        else:
            self.agent_list = [agent]
            self.agent_hub = None

        # 配置用户信息
        user_name = chatbot_config.get('user.name', 'user')
        self.user_config = {
            'name': user_name,
            'avatar': chatbot_config.get(
                'user.avatar',
                get_avatar_image(user_name),
            ),
        }

        # 配置代理列表信息
        self.agent_config_list = [{
            'name': agent.name,
            'avatar': chatbot_config.get(
                'agent.avatar',
                get_avatar_image(agent.name),
            ),
            'description': agent.description or "I'm a helpful assistant.",
        } for agent in self.agent_list]

        # 配置界面其他元素
        self.input_placeholder = chatbot_config.get('input.placeholder', '跟我聊聊吧～')
        self.prompt_suggestions = chatbot_config.get('prompt.suggestions', [])
        self.verbose = chatbot_config.get('verbose', False)

    """
    Run the chatbot.

    Args:
        messages: The chat history.
    """

    def run(self,
            messages: List[Message] = None,
            share: bool = False,
            server_name: str = None,
            server_port: int = None,
            concurrency_limit: int = 10,
            enable_mention: bool = False,
            **kwargs):
        # 保存额外的运行参数到实例变量
        self.run_kwargs = kwargs

        # 导入 Gradio 相关依赖
        from qwen_agent.gui.gradio_dep import gr, mgr, ms

        # 创建自定义主题，使用蓝色作为主色调，无圆角设计
        customTheme = gr.themes.Default(
            primary_hue=gr.themes.utils.colors.blue,
            radius_size=gr.themes.utils.sizes.radius_none,
        )

        # 创建 Gradio Blocks 应用，加载自定义 CSS 和主题
        with gr.Blocks(
                css=os.path.join(os.path.dirname(__file__), 'assets/appBot.css'),
                theme=customTheme,
        ) as demo:
            # 初始化聊天历史状态
            history = gr.State([])
            # 使用 ModelScope UI 组件
            with ms.Application():
                # 创建横向布局容器
                with gr.Row(elem_classes='container'):
                    # 左侧主内容区域（占 4/5 宽度）
                    with gr.Column(scale=4):
                        # 创建聊天机器人组件，配置头像、高度、LaTeX 渲染等
                        chatbot = mgr.Chatbot(value=convert_history_to_chatbot(messages=messages),
                                              avatar_images=[
                                                  self.user_config,
                                                  self.agent_config_list,
                                              ],
                                              height=850,
                                              avatar_image_width=80,
                                              flushing=False,
                                              show_copy_button=True,
                                              latex_delimiters=[{
                                                  'left': '\\(',
                                                  'right': '\\)',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{equation}',
                                                  'right': '\\end{equation}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{align}',
                                                  'right': '\\end{align}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{alignat}',
                                                  'right': '\\end{alignat}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{gather}',
                                                  'right': '\\end{gather}',
                                                  'display': True
                                              }, {
                                                  'left': '\\begin{CD}',
                                                  'right': '\\end{CD}',
                                                  'display': True
                                              }, {
                                                  'left': '\\[',
                                                  'right': '\\]',
                                                  'display': True
                                              }])

                        # 创建多模态输入框（支持文本、图片、文件等）
                        input = mgr.MultimodalInput(placeholder=self.input_placeholder,)
                        # 创建音频输入组件（麦克风）
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath"
                        )

                    # 右侧边栏区域（占 1/5 宽度）
                    with gr.Column(scale=1):
                        # 如果有多个代理，显示代理选择下拉框
                        if len(self.agent_list) > 1:
                            agent_selector = gr.Dropdown(
                                [(agent.name, i) for i, agent in enumerate(self.agent_list)],
                                label='Agents',
                                info='选择一个Agent',
                                value=0,
                                interactive=True,
                            )

                        # 创建代理信息展示区块
                        agent_info_block = self._create_agent_info_block()

                        # 创建代理插件展示区块
                        agent_plugins_block = self._create_agent_plugins_block()

                        # 如果配置了推荐问题，显示推荐对话示例
                        if self.prompt_suggestions:
                            gr.Examples(
                                label='推荐对话',
                                examples=self.prompt_suggestions,
                                inputs=[input],
                            )

                    # 为代理选择器绑定变更事件
                    if len(self.agent_list) > 1:
                        agent_selector.change(
                            fn=self.change_agent,
                            inputs=[agent_selector],
                            outputs=[agent_selector, agent_info_block, agent_plugins_block],
                            queue=False,
                        )

                    # 为输入框绑定提交事件，处理用户输入
                    input_promise = input.submit(
                        fn=self.add_text,
                        inputs=[input, audio_input, chatbot, history],
                        outputs=[input, audio_input, chatbot, history],
                        queue=False,
                    )

                    # 如果是多代理模式且启用了 @ 提及功能，添加额外的处理步骤
                    if len(self.agent_list) > 1 and enable_mention:
                        input_promise = input_promise.then(
                            self.add_mention,
                            [chatbot, agent_selector],
                            [chatbot, agent_selector],
                        ).then(
                            self.agent_run,
                            [chatbot, history, agent_selector],
                            [chatbot, history, agent_selector],
                        )
                    # 否则直接运行代理响应
                    else:
                        input_promise = input_promise.then(
                            self.agent_run,
                            [chatbot, history],
                            [chatbot, history],
                        )

                    # 输入处理完成后，重置输入框状态
                    input_promise.then(self.flushed, None, [input])

            # 加载页面
            demo.load(None)

        # 启动 Gradio 服务，配置并发数和服务器信息
        demo.queue(default_concurrency_limit=concurrency_limit).launch(share=share,
                                                                       server_name=server_name,
                                                                       server_port=server_port)

    @log_execution
    def change_agent(self, agent_selector):
        yield agent_selector, self._create_agent_info_block(agent_selector), self._create_agent_plugins_block(
            agent_selector)

    @log_execution
    def add_text(self, _input, _audio_input, _chatbot, _history):
        _history.append({
            ROLE: USER,
            CONTENT: [{
                'text': _input.text
            }],
        })

        if self.user_config[NAME]:
            _history[-1][NAME] = self.user_config[NAME]
        
        # if got audio from microphone, append it to the multimodal inputs
        if _audio_input:
            from qwen_agent.gui.gradio_dep import gr, mgr, ms
            audio_input_file = gr.data_classes.FileData(path=_audio_input, mime_type="audio/wav")
            _input.files.append(audio_input_file)

        if _input.files:
            for file in _input.files:
                if file.mime_type.startswith('image/'):
                    _history[-1][CONTENT].append({IMAGE: 'file://' + file.path})
                elif file.mime_type.startswith('audio/'):
                    _history[-1][CONTENT].append({AUDIO: 'file://' + file.path})
                elif file.mime_type.startswith('video/'):
                    _history[-1][CONTENT].append({VIDEO: 'file://' + file.path})
                else:
                    _history[-1][CONTENT].append({FILE: file.path})

        _chatbot.append([_input, None])

        from qwen_agent.gui.gradio_dep import gr

        yield gr.update(interactive=False, value=None), None, _chatbot, _history

    @log_execution
    def add_mention(self, _chatbot, _agent_selector):
        if len(self.agent_list) == 1:
            yield _chatbot, _agent_selector

        query = _chatbot[-1][0].text
        match = re.search(r'@\w+\b', query)
        if match:
            _agent_selector = self._get_agent_index_by_name(match.group()[1:])

        agent_name = self.agent_list[_agent_selector].name

        if ('@' + agent_name) not in query and self.agent_hub is None:
            _chatbot[-1][0].text = '@' + agent_name + ' ' + query

        yield _chatbot, _agent_selector
    @log_execution
    def agent_run(self, _chatbot, _history, _agent_selector=None):
        # 如果启用了详细日志模式，记录传入的对话历史用于调试
        if self.verbose:
            logger.info('agent_run input:\n' + pprint.pformat(_history, indent=2))

        # 计算输入消息的数量（聊天窗口中用户已发送的消息气泡数）
        num_input_bubbles = len(_chatbot) - 1
        # 初始化输出消息气泡数量为1（当前正在生成的回复）
        num_output_bubbles = 1
        # 初始化最后一个聊天气泡的代理回复列表，为每个代理创建一个None占位符
        _chatbot[-1][1] = [None for _ in range(len(self.agent_list))]

        # 根据代理选择器确定要运行的代理，默认选择第一个代理（索引0）
        agent_runner = self.agent_list[_agent_selector or 0]
        # 如果使用了多代理中心（MultiAgentHub），则使用中心作为代理运行器
        if self.agent_hub:
            agent_runner = self.agent_hub
        # 初始化响应列表，用于存储代理返回的消息
        responses = []
        # 流式调用代理的run方法，逐步获取代理生成的响应片段
        for responses in agent_runner.run(_history, **self.run_kwargs):
            # 如果本次没有返回任何响应，跳过继续等待
            if not responses:
                continue
            # 检查最后一条响应的内容是否为"等待用户输入"状态（表示代理需要用户进一步交互）
            if responses[-1][CONTENT] == PENDING_USER_INPUT:
                # 记录中断日志，说明代理正在等待用户的下一步输入
                logger.info('Interrupted. Waiting for user input!')
                # 跳出流式循环，停止生成
                break

            # 将函数调用（function call）格式的响应转换为可读文本格式
            display_responses = convert_fncall_to_text(responses)
            # 如果转换后没有可显示的内容，跳过
            if not display_responses:
                continue
            # 如果最后一条响应的内容为空，跳过
            if display_responses[-1][CONTENT] is None:
                continue

            # 当需要显示的响应数量超过当前输出气泡数量时，创建新的聊天气泡
            while len(display_responses) > num_output_bubbles:
                # 在聊天窗口中添加一个新的空白消息对[用户消息, 代理回复]
                _chatbot.append([None, None])
                # 为新气泡初始化所有代理的回复占位符列表
                _chatbot[-1][1] = [None for _ in range(len(self.agent_list))]
                # 增加输出气泡计数器
                num_output_bubbles += 1

            # 断言：确保输出气泡数量与需要显示的响应数量一致
            assert num_output_bubbles == len(display_responses)
            # 断言：确保总气泡数等于输入气泡数加输出气泡数
            assert num_input_bubbles + num_output_bubbles == len(_chatbot)

            # 遍历每条需要显示的响应消息
            for i, rsp in enumerate(display_responses):
                # 根据响应中的代理名称查找其在代理列表中的索引位置
                agent_index = self._get_agent_index_by_name(rsp[NAME])
                # 将响应内容填充到对应聊天气泡的对应代理位置
                _chatbot[num_input_bubbles + i][1][agent_index] = rsp[CONTENT]

            # 如果有多个代理，更新代理选择器为当前响应的代理
            if len(self.agent_list) > 1:
                _agent_selector = agent_index

            # 根据是否有代理选择器，决定yield返回的参数数量
            if _agent_selector is not None:
                # 返回更新的聊天界面、历史记录和代理选择器状态
                yield _chatbot, _history, _agent_selector
            else:
                # 返回更新的聊天界面和历史记录
                yield _chatbot, _history

        # 流式生成结束后，将所有非"等待用户输入"的响应添加到对话历史中
        if responses:
            _history.extend([res for res in responses if res[CONTENT] != PENDING_USER_INPUT])

        # 最后一次yield，确保最终状态被正确传递
        if _agent_selector is not None:
            yield _chatbot, _history, _agent_selector
        else:
            yield _chatbot, _history

        # 如果启用了详细日志模式，记录代理的最终响应内容用于调试
        if self.verbose:
            logger.info('agent_run response:\n' + pprint.pformat(responses, indent=2))


    @log_execution
    def flushed(self):
        from qwen_agent.gui.gradio_dep import gr

        return gr.update(interactive=True)

    @log_execution
    def _get_agent_index_by_name(self, agent_name):
        if agent_name is None:
            return 0

        try:
            agent_name = agent_name.strip()
            for i, agent in enumerate(self.agent_list):
                if agent.name == agent_name:
                    return i
            return 0
        except Exception:
            print_traceback()
            return 0

    @log_execution
    def _create_agent_info_block(self, agent_index=0):
        from qwen_agent.gui.gradio_dep import gr

        agent_config_interactive = self.agent_config_list[agent_index]

        return gr.HTML(
            format_cover_html(
                bot_name=agent_config_interactive['name'],
                bot_description=agent_config_interactive['description'],
                bot_avatar=agent_config_interactive['avatar'],
            ))

    @log_execution
    def _create_agent_plugins_block(self, agent_index=0):
        from qwen_agent.gui.gradio_dep import gr

        agent_interactive = self.agent_list[agent_index]

        if agent_interactive.function_map:
            capabilities = [key for key in agent_interactive.function_map.keys()]
            return gr.CheckboxGroup(
                label='插件',
                value=capabilities,
                choices=capabilities,
                interactive=False,
            )

        else:
            return gr.CheckboxGroup(
                label='插件',
                value=[],
                choices=[],
                interactive=False,
            )
