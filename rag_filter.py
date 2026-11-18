"""检索增强：查询改写 + 岗位类型过滤

解决两个实际问题：
    1. 「15K 以上大模型岗」会召回「大模型运营（商城方向）」——字面匹配但语义跑偏
    2. 用户说「技术岗」但查询里没这个词，模型不知道这个隐含意图

用法：
    from rag_filter import rewrite_query, filter_by_title, EXCLUDED_KEYWORDS
"""
import re

# 非技术岗关键词（命中就排掉）
EXCLUDED_KEYWORDS = [
    "运营", "销售", "商务", "市场", "交付", "项目经理", "实施",
    "讲师", "培训", "咨询", "客服", "售前", "售后",
    "标注", "训练师", "数据标注", "评测专员",
]

# 技术岗正向关键词（加分项，用于排序时可参考）
TECH_KEYWORDS = [
    "开发", "工程师", "算法", "架构", "研发", "研究",
    "AI", "LLM", "大模型", "NLP", "CV", "多模态", "Agent", "RAG",
    "Python", "Java", "C++", "Go",
]

# 同义扩展：把口语化的词映射成更精确的检索词
QUERY_EXPANSIONS = {
    "技术岗": "开发 工程师 算法",
    "开发岗": "开发 工程师",
    "算法岗": "算法 工程师",
    "搞技术": "开发 工程师 算法",
    "写代码": "开发 工程师",
    "AI岗": "大模型 算法 AI",
    "入门": "实习 应届 经验不限",
    "应届": "应届 校招 经验不限",
    "实习": "实习 日常实习",
}


def rewrite_query(query: str) -> str:
    """把口语化的查询改写成更精确的检索语句。

    例：把「有什么技术岗」→「有什么技术岗 开发 工程师 算法」
    """
    extra = []
    for pattern, expansion in QUERY_EXPANSIONS.items():
        if pattern in query:
            extra.append(expansion)
    if extra:
        return query + " " + " ".join(extra)
    return query


def should_exclude(title: str) -> bool:
    """判断这个岗位是否应该被过滤掉（非技术岗）"""
    return any(kw in title for kw in EXCLUDED_KEYWORDS)


def filter_by_title(hits, id2chunk, min_keep=3):
    """过滤掉非技术岗，但保底留下 min_keep 条。

    hits: [(chunk_id, score), ...]
    返回: 过滤后的 hits
    """
    kept = []
    dropped = []
    for cid, score in hits:
        c = id2chunk.get(cid)
        if not c:
            continue
        if should_exclude(c["job_title"]):
            dropped.append((cid, c["job_title"]))
        else:
            kept.append((cid, score))

    # 保底：如果过滤后太少，把被丢的补回来（避免空结果）
    if len(kept) < min_keep:
        kept = hits[:min_keep]

    return kept, dropped


def extract_salary_floor(query: str):
    """从查询里提取薪资下限（如 '15K' → 15000）

    返回 None 表示没提取到
    """
    # 匹配 15K / 15k / 1.5万 / 15000
    m = re.search(r"(\d+(?:\.\d+)?)\s*[kK]", query)
    if m:
        return float(m.group(1)) * 1000
    m = re.search(r"(\d+(?:\.\d+)?)\s*万", query)
    if m:
        return float(m.group(1)) * 10000
    return None


def parse_salary_range(salary_str: str):
    """解析薪资字符串，返回 (下限, 上限) 元/月

    能处理：
        15-30K·15薪  → (15000, 30000)
        150-200元/天 → (4500, 6000) 按 30 天估算
        6-11K        → (6000, 11000)
    """
    if not salary_str:
        return None

    # K 为单位
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*[kK]", salary_str)
    if m:
        return (float(m.group(1)) * 1000, float(m.group(2)) * 1000)

    # 元/天
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*元/天", salary_str)
    if m:
        return (int(m.group(1)) * 30, int(m.group(2)) * 30)

    # 万/月
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*万", salary_str)
    if m:
        return (float(m.group(1)) * 10000, float(m.group(2)) * 10000)

    return None


def filter_by_salary(hits, id2chunk, floor, strict=True):
    """按薪资下限过滤。

    strict=True（默认）：只有「下限 >= floor」才保留
        搜索「15K以上」时，10-15K 会被排除（因为下限是 10K）
    strict=False：只要「上限 >= floor」就保留（宽松模式）
    """
    if floor is None:
        return hits, []
    kept, dropped = [], []
    for cid, score in hits:
        c = id2chunk.get(cid)
        if not c:
            continue
        rng = parse_salary_range(c.get("salary", ""))
        if rng is None:
            # 薪资解析不出，保守保留
            kept.append((cid, score))
        elif (rng[0] >= floor) if strict else (rng[1] >= floor):
            kept.append((cid, score))
        else:
            dropped.append((cid, c["job_title"], c["salary"]))
    return kept, dropped
