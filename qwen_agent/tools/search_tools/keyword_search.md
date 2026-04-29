## `search` 方法关键流程说明

### 方法签名
```python
def search(self, query: str, docs: List[Record], max_ref_token: int = DEFAULT_MAX_REF_TOKEN) -> list
```


### 核心目的
执行关键词搜索，从文档列表中找出与查询最相关的文档片段，并控制返回结果的token数量。

---

### 关键流程步骤

#### **步骤1：计算相关性分数并排序**
```python
chunk_and_score = self.sort_by_scores(query=query, docs=docs)
```

- **调用方法**：`sort_by_scores()`（第64-100行）
- **作用**：
  - 通过 `parse_keyword()` 解析查询文本提取关键词
  - 收集所有文档的所有片段
  - 使用 **BM25算法** 计算每个片段与查询的相关性分数
  - 返回按分数降序排列的 `(来源, 片段ID, 分数)` 元组列表
- **目的**：为所有文档片段打分并排序，找出最相关的片段

#### **步骤2：处理无匹配情况**
```python
if not chunk_and_score:
    return self._get_the_front_part(docs, max_ref_token)
```

- **调用方法**：`_get_the_front_part()`（继承自父类 `BaseSearch`）
- **作用**：当无法提取关键词时（如总结类查询），直接返回文档的前部内容
- **目的**：兜底策略，确保即使没有关键词匹配也能返回内容

#### **步骤3：获取最高相似度分数**
```python
max_sims = chunk_and_score[0][-1]
```

- **作用**：提取排序后第一个结果（最高分）的分数值
- **目的**：判断是否存在有效匹配（分数是否为0）

#### **步骤4：根据匹配有效性返回结果**
```python
if max_sims != 0:
    return super().get_topk(chunk_and_score=chunk_and_score, docs=docs, max_ref_token=max_ref_token)
else:
    return self._get_the_front_part(docs, max_ref_token)
```


**情况A：存在有效匹配（`max_sims != 0`）**
- **调用方法**：`super().get_topk()`（父类 `BaseSearch` 的方法）
- **作用**：
  - 根据 `max_ref_token` 限制，从高分到低分选取文档片段
  - 确保返回内容的token数不超过限制
- **目的**：返回最相关的Top-K个文档片段

**情况B：无有效匹配（`max_sims == 0`）**
- **调用方法**：`_get_the_front_part()`
- **作用**：返回文档前部内容
- **目的**：兜底策略，当BM25评分全为0时提供基础内容

---

### 方法调用链路图

```
search()
├── sort_by_scores()                    # 计算BM25相关性分数
│   ├── parse_keyword()                 # 解析查询提取关键词
│   │   └── split_text_into_keywords()  # 分词和过滤
│   └── BM25Okapi.get_scores()          # BM25算法打分
│
├── _get_the_front_part()               # [兜底] 返回文档前部
│   └── (父类BaseSearch方法)
│
└── get_topk()                          # [主路径] 返回Top-K相关片段
    └── (父类BaseSearch方法)
```


---

### 设计意图总结

1. **主要路径**：使用BM25算法进行关键词匹配，返回最相关的文档片段
2. **容错机制**：三种兜底场景
   - 无法提取关键词 → 返回文档前部
   - BM25评分全为0 → 返回文档前部
   - JSON解析失败 → 降级为普通文本分词
3. **资源控制**：通过 `max_ref_token` 限制返回内容的token数量，避免超出上下文窗口