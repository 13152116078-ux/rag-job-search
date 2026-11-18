# RAG 岗位检索系统 · 操作手册

> 面向：你自己用 / 面试时演示
> 环境：Windows + DSH 自带 Python

---

## 一、先认清楚：这套东西由哪些文件组成

| 文件 | 作用 | 什么时候用 |
|---|---|---|
| `jobs.jsonl` | 原始数据（243 条岗位） | 数据源，不用手动动 |
| `boss-ai-jobs.csv` | 同上，Excel 版 | 想用 Excel 看时 |
| `chunk-jobs.py` | 把岗位切成 chunk | **只在换数据时跑** |
| `chunks.jsonl` | 切好的 243 个 chunk | 中间产物 |
| `build-index.py` | 向量化 + 入库 | **只在换数据时跑** |
| `chroma_db/` | 向量库（243×1024 维） | 中间产物，别删 |
| `search.py` | 纯向量检索 | 演示用（对比基线） |
| `hybrid_search.py` | 混合检索（向量+BM25） | 演示用（对比） |
| `rerank_search.py` | 混合 + CrossEncoder 重排 | 演示用（完整检索） |
| `eval_quick.py` | 快速评测（自检索） | 改代码后验证 |
| `eval_rerank.py` | 加重排的评测 | 改代码后验证 |
| `rag_chat.py` | **完整 RAG：检索→重排→生成** | **日常主力** |
| `rag_filter.py` | 查询改写 + 岗位/薪资过滤 | 被 rag_chat 自动调用 |

**一句话记住**：平时只用 `rag_chat.py`，其他都是它的零件或实验品。

---

## 二、日常使用（90% 的场景）

### 场景 1：问一个问题，拿到答案

```powershell
cd "C:\Users\BYS\Documents\deepseek-harness\default-workspace"

# 设置模型路径（每次开新终端都要设）
$py = "C:\Users\BYS\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# 问问题
& $py rag_chat.py "西安有哪些薪资15K以上的大模型技术岗"
```

**会看到什么**：
```
原始查询: 西安有哪些薪资15K以上的大模型技术岗
改写后:   西安有哪些薪资15K以上的大模型技术岗 开发 工程师 算法 应届 校招 经验不限
薪资下限: 15000 元/月

[1/4] 检索... 20 条
[2/4] 重排... Top 5
  过滤掉 3 条（薪资/岗位类型不符）
[3/4] 组装 Prompt... 1436 字符
[4/4] 无 API key 或 --no-llm：输出 Prompt

（下面是拼好的 Prompt，复制到 DeepSeek 网页版即可）
```

### 场景 2：有 API key，直接出答案

```powershell
$env:DEEPSEEK_API_KEY = "sk-你的key"
& $py rag_chat.py "西安有哪些15K以上的大模型岗"
```

**会看到**：模型生成的答案 + 末尾附「引用来源」列表。

### 场景 3：只要检索结果，不要生成

```powershell
# 看检索排出来什么（不拼 prompt）
& $py rerank_search.py "需要 PyTorch 的岗位" --compare
```

`--compare` 会同时显示**重排前**和**重排后**的对比，适合演示。

---

## 三、完整流程（换数据时跑一次）

假设你新抓了一批数据，或者想重建索引：

```powershell
cd "C:\Users\BYS\Documents\deepseek-harness\default-workspace"
$py = "C:\Users\BYS\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"

# 步骤 1：切 chunk（把 jobs.jsonl 变成 chunks.jsonl）
& $py chunk-jobs.py

# 步骤 2：向量化 + 入库（第一次要下载 2GB 模型，之后走本地缓存）
$env:HF_HUB_OFFLINE = ""
$env:HF_ENDPOINT = "https://hf-mirror.com"
& $py build-index.py

# 步骤 3：验证（跑评测看 Recall@5）
$env:HF_HUB_OFFLINE = "1"
& $py eval_quick.py --sample 50
```

**注意**：
- 步骤 2 **必须联网**（要下 BPE 文件），不能设 `HF_HUB_OFFLINE=1`
- 步骤 1 和 3 **必须离线**，设 `HF_HUB_OFFLINE=1` 否则会卡在联网检查

---

## 四、输出文件在哪

| 场景 | 输出 |
|---|---|
| `rag_chat.py --save xxx.txt` | Prompt 存到指定文件 |
| `eval_quick.py` | `eval_result.json`（Recall 结果） |
| `eval_rerank.py` | `eval_rerank_result.json` |
| `rag_chat.py` 调 API | 直接打印到终端 |

---

## 五、所有可用参数

### rag_chat.py

```powershell
& $py rag_chat.py "查询词" [参数]

参数：
  --top 5           最终给模型几条资料（默认 5）
  --recall 20       粗召回多少条（默认 20）
  --no-llm          只输出 prompt，不调 API
  --no-filter       关闭岗位/薪资过滤
  --save 文件名     把 prompt 存到文件
```

### eval_quick.py / eval_rerank.py

```powershell
参数：
  --sample 50       评测多少条 query（默认 50 / 20）
  --topk 5          评估 Recall@K（默认 5）
  --recall 20       每路召回多少条
```

---

## 六、常见问题

### Q1：报 `ModuleNotFoundError: No module named 'xxx'`

**原因**：用错了 python。必须用 DSH 自带的那个：
```
C:\Users\BYS\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe
```
不是系统 PATH 里的 `python`。

### Q2：卡住不动，CPU 占用为 0

**原因**：多半是 `HF_HUB_OFFLINE` 设置不对，程序在等网络超时。
- `build-index.py` → **不要**设 `HF_HUB_OFFLINE`
- 其他脚本 → **要**设 `HF_HUB_OFFLINE=1`

### Q3：报 `Pooling.__init__() missing 1 required positional argument`

**原因**：`sentence-transformers` 和 `transformers` 版本不兼容。
**已验证可用的组合**：
```
torch 2.15.0+cpu
transformers 4.49.0
sentence-transformers 3.4.1
chromadb 1.5.9
tiktoken 0.14.0
```

### Q4：检索出来的岗位不对（比如搜技术岗出来运营岗）

1. 检查是否设了 `--no-filter`（设了就不过滤了）
2. 看输出里的「改写后」——如果没加技术词，说明你的查询里没有触发关键词
3. 在 `rag_filter.py` 的 `QUERY_EXPANSIONS` 里加你常用的说法

### Q5：想看某个具体岗位的完整信息

```powershell
# 用标题当查询词
& $py search.py "大模型算法工程师" --top 3
```

---

## 七、面试演示建议（5 分钟版本）

**第 1 分钟**：讲架构
> 「这是一个 RAG 岗位检索系统。数据是我从 Boss 直聘抓的 243 条真实岗位，走的是 向量召回 + BM25 召回 → RRF 融合 → CrossEncoder 重排 → 拼 Prompt 生成 这条链路。」

**第 2 分钟**：演示纯检索
```powershell
& $py rerank_search.py "需要 PyTorch 的大模型岗" --compare
```
> 「左边是重排前，右边是重排后。重排把『大模型运营』这类字面相关但语义跑偏的岗位挤出去了。」

**第 3 分钟**：讲评测
> 「我做了 Recall@5 评测：纯向量 92%，混合 94%，加重排 100%（但样本只有 20 条，1 条差异就是 5 个点，所以这个数字要谨慎看）。」

**第 4 分钟**：讲遇到的问题
> 「最花时间的不是写代码，是环境。我踩了 5 个坑：pip 临时目录、tqdm 管道阻塞、transformers 版本冲突、tiktoken 缺失、离线模式反噬。其中 tqdm 那个最隐蔽——进程 CPU 为 0 但没报错，是因为进度条输出到管道阻塞了。」

**第 5 分钟**：讲下一步
> 「当前最大的短板是数据只有 243 条，评测用的是自检索（用标题查自己），比真实提问容易。下一步应该用真实用户 query 建评测集，才能真正验证效果。」

---

## 八、一句话总结

**日常**：`python rag_chat.py "你的问题"`
**换数据**：`chunk-jobs.py` → `build-index.py`（联网）→ `eval_quick.py`（离线）
**演示对比**：`rerank_search.py "查询" --compare`
