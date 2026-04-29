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

from typing import Dict, List, Optional, Tuple

from qwen_agent.settings import DEFAULT_RAG_SEARCHERS
from qwen_agent.tools.base import TOOL_REGISTRY, register_tool
from qwen_agent.tools.doc_parser import Record
from qwen_agent.tools.search_tools.base_search import BaseSearch
from qwen_agent.tools.search_tools.front_page_search import POSITIVE_INFINITY


@register_tool('hybrid_search')
class HybridSearch(BaseSearch):
    # 初始化混合搜索对象
    def __init__(self, cfg: Optional[Dict] = None):
        # 调用父类初始化方法
        super().__init__(cfg)
        # 从配置中获取RAG搜索器列表，如果未配置则使用默认值
        self.rag_searchers = self.cfg.get('rag_searchers', DEFAULT_RAG_SEARCHERS)

        # 检查当前混合搜索工具名称是否出现在子搜索器列表中，避免递归调用
        if self.name in self.rag_searchers:
            raise ValueError(f'{self.name} can not be in `rag_searchers` = {self.rag_searchers}')
        # 创建所有子搜索器实例
        self.search_objs = [TOOL_REGISTRY[name](cfg) for name in self.rag_searchers]

    @log_execution
    # 根据多个搜索器的结果对文档块进行排序
    def sort_by_scores(self, query: str, docs: List[Record], **kwargs) -> List[Tuple[str, int, float]]:
        # 存储每个搜索器返回的排序结果
        chunk_and_score_list = []
        # 遍历所有子搜索器，分别执行排序
        # KeywordSearch
        # FrontPageSearch
        for s_obj in self.search_objs:
            # 调用每个搜索器的排序方法并收集结果
            chunk_and_score_list.append(s_obj.sort_by_scores(query=query, docs=docs, **kwargs))

        # 构建文档块的分数映射表，初始化为0
        chunk_score_map = {}
        for doc in docs:
            # 为每个文档的所有块初始化分数列表
            chunk_score_map[doc.url] = [0] * len(doc.raw)

        # 合并所有搜索器的排序结果
        for chunk_and_score in chunk_and_score_list:
            for i in range(len(chunk_and_score)):
                # 提取文档ID、块ID和分数
                doc_id = chunk_and_score[i][0]
                chunk_id = chunk_and_score[i][1]
                score = chunk_and_score[i][2]
                # 如果分数为正无穷，直接设置为正无穷（表示完全匹配）
                if score == POSITIVE_INFINITY:
                    chunk_score_map[doc_id][chunk_id] = POSITIVE_INFINITY
                else:
                    # TODO: This needs to be adjusted for performance
                    # 使用倒数排名融合算法（RRF）累加分数
                    chunk_score_map[doc_id][chunk_id] += 1 / (i + 1 + 60)

        # 将所有文档块的分数汇总到一个列表中
        all_chunk_and_score = []
        for k, v in chunk_score_map.items():
            for i, x in enumerate(v):
                all_chunk_and_score.append((k, i, x))
        # 按分数降序排序所有文档块
        all_chunk_and_score.sort(key=lambda item: item[2], reverse=True)

        # 返回排序后的文档块列表
        return all_chunk_and_score

# ... existing code ...
