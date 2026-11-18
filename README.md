# 岗位检索助手 · RAG Job Search

> 一个面向求职场景的 RAG（检索增强生成）系统。抓取真实岗位数据，用**混合检索 + 重排**找到最相关的岗位，并基于简历做**个性化匹配**。

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.162-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 一、项目背景

找工作时有两个痛点：

1. **招聘网站的搜索是关键词匹配**，搜「大模型」会混进来一堆标注岗、销售岗
2. **不知道自己的简历和哪个岗位最匹配**，只能一个个点开看

这个项目用 RAG 的思路解决：把岗位数据向量化，用语义 + 关键词双路检索，再用 CrossEncoder 精排，最后让 LLM 基于检索结果回答。同时用技能词表做简历与岗位的匹配打分。

---

## 二、技术架构

```
岗位数据 (243 条)
     ↓  结构化拼接
自然语言文档
     ↓  切片（chunk_size=400, overlap=80）
243 个 chunk
     ↓  BGE-M3 向量化（1024 维）
   Chroma 向量库
     ↓
┌──────────────────────────────┐
│  用户查询                     │
│    ↓                         │
│  查询改写（技术岗→开发/算法）  │
│    ↓                         │
│  ┌─────────┬─────────┐       │
│  │ 向量检索 │ BM25检索 │       │
│  │  Top-20 │  Top-20 │       │
│  └────┬────┴────┬────┘       │
│       └──┬──────┘            │
│         RRF 融合              │
│          ↓                   │
│    CrossEncoder 重排          │
│          ↓                   │
│    岗位/薪资过滤               │
│          ↓                   │
│        Top-5                 │
└──────────────────────────────┘
     ↓
Prompt 组装（带引用编号）
     ↓
LLM 生成 / 直接输出 Prompt
```

### 技术选型理由

| 组件 | 选择 | 为什么 |
|---|---|---|
| 嵌入模型 | **BGE-M3** | 中文效果好、支持长文本（8192 token）、本地可跑 |
| 向量库 | **Chroma** | 嵌入式、持久化、支持 metadata 过滤；243 条数据不需要 Milvus |
| 关键词检索 | **BM25 + jieba** | 补向量检索对专有名词（PyTorch、RAG）不敏感的短板 |
| 融合策略 | **RRF（k=60）** | 向量分数（0-1）和 BM25 分数（0-20+）量纲不同，RRF 只看排名不看分值 |
| 重排模型 | **bge-reranker-v2-m3** | Cross-encoder 让 query 和 doc 充分交互，比 bi-encoder 准得多 |
| 服务框架 | **FastAPI** | 自动生成 OpenAPI 文档、异步、轻量 |

---

## 三、核心亮点

### 1. 混合检索解决「语义 vs 精确」的冲突

纯向量检索对「PyTorch」「RAG」这类专有名词不敏感——它们语义上可能匹配到一堆泛泛的「模型」岗位。BM25 补上了精确匹配，RRF 融合后两者互补。

```python
# RRF 融合：只看排名，规避量纲问题
fused[doc_id] += 1.0 / (k + rank)   # k=60
```

### 2. 两级检索架构（召回 → 重排）

- **Bi-encoder** 粗召回：查询和文档分别编码，快，可预计算
- **Cross-encoder** 精排：把 (query, doc) 拼起来喂模型，慢但准

用便宜的粗召回把候选压到 20 条，再用贵的模型精挑 5 条。

### 3. 可量化的评估

构建 150 条评测集，对比不同方案：

| 方案 | Recall@5 (50条样本) | Recall@5 (20条样本) |
|---|---|---|
| 纯向量检索 | 92.0% | 95.0% |
| + BM25 混合 | 94.0% | 95.0% |
| + CrossEncoder 重排 | — | **100.0%** |

> ⚠️ 说明：当前评测用的是「自检索」（用岗位标题查自己），比真实提问容易。真实查询下召回率会低一些。这是已知局限，下一步计划用真实用户 query 重建评测集。

### 4. 三层幻觉控制

1. **Prompt 约束**：要求「资料没有就说没有」
2. **相似度阈值**：Top-1 低于阈值时直接拒答
3. **强制引用**：每句结论必须标 `[1]` `[2]`，用户可核对

### 5. 简历匹配

从简历抽取技能（覆盖 100+ 技能词），与岗位要求做加权匹配：

```
总分 = 技能重合度 × 0.7 + 硬性条件（学历/经验） × 0.3
```

输出「你已具备的技能」和「岗位还要求的技能」，让用户知道该补什么。

---

## 四、快速开始

### 环境要求

- Python 3.10+
- 约 5GB 磁盘空间（模型缓存）

### 安装

```bash
pip install -r requirements.txt
```

### 准备数据

```bash
# 1. 切分 chunk
python chunk-jobs.py

# 2. 向量化 + 入库（首次会下载约 2GB 模型）
python build-index.py
```

### 启动服务

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

打开浏览器访问 **http://127.0.0.1:8000/** —— 中文界面，含「搜岗位」和「简历匹配」两个功能。

### 命令行使用

```bash
# 只检索
python rerank_search.py "需要 PyTorch 的岗位" --compare

# 检索 + 生成 Prompt
python rag_chat.py "西安有哪些 15K 以上的大模型岗"

# 有 API key 时直接出答案
export DEEPSEEK_API_KEY="sk-xxx"
python rag_chat.py "西安有哪些 15K 以上的大模型岗"
```

---

## 五、项目结构

```
.
├── api.py                  # FastAPI 服务（/search /match /chat）
├── web/
│   └── index.html          # 中文前端（搜索 + 简历匹配）
├── rag_filter.py           # 查询改写 + 岗位/薪资过滤
├── resume_match.py         # 简历技能抽取 + 匹配打分
├── rag_chat.py             # 命令行版完整 RAG
├── hybrid_search.py        # 混合检索（向量 + BM25 + RRF）
├── rerank_search.py        # 混合检索 + 重排
├── search.py               # 纯向量检索（基线）
├── build-index.py          # 向量化 + 入库
├── chunk-jobs.py           # 数据切片
├── eval_quick.py           # 快速评测（自检索）
├── eval_rerank.py          # 重排效果评测
├── jobs.jsonl              # 原始数据（243 条岗位）
├── chunks.jsonl            # 切片结果（243 个 chunk）
├── chroma_db/              # 向量库
└── docs/
    ├── RAG项目方案.md
    ├── RAG实施手册-目的做法面试题.md
    ├── RAG操作手册.md
    └── 傻瓜教程-小白版.md
```

---

## 六、踩过的坑（工程笔记）

| 问题 | 现象 | 解决 |
|---|---|---|
| pip 临时目录 | `Failed to remove temp directory` | 把 `TEMP` 指向工作区 + `--user` |
| tqdm 进度条 | 进程 CPU 为 0，卡 20 分钟 | 关掉 `show_progress_bar`（管道阻塞） |
| transformers 版本冲突 | `Pooling.__init__() missing embedding_dimension` | 降级到 `transformers==4.49.0` + `sentence-transformers==3.4.1` |
| tiktoken 缺失 | `No module named 'tiktoken'` | 单独安装 |
| 离线模式反噬 | `'NoneType' has no attribute 'encode'` | `build-index.py` 不能设 `HF_HUB_OFFLINE` |
| JSON 字段缺失 | `KeyError: 'experience'` | `chunks.jsonl` 里没有该字段，改用 `.get()` 兜底 |

---

## 七、已知局限与后续计划

**局限**：

1. 数据量小（243 条），评测用的是自检索，偏乐观
2. 只支持单轮查询，不支持多轮对话
3. 简历技能抽取用的是词表匹配，没有用 NER 模型

**计划**：

- [ ] 用真实用户 query 重建评测集（150+ 条）
- [ ] 支持多轮对话（记忆上下文）
- [ ] 接入 LLM 做查询改写（替代关键词表）
- [ ] 岗位详情页 + 收藏功能

---

## 八、License

MIT
