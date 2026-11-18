"""阶段 5：RAG 生成（带引用）

流程：查询 → 混合召回 → 重排 → 拼 Prompt → 调 LLM / 输出 Prompt

两种模式：
    1. 有 DEEPSEEK_API_KEY 环境变量 → 直接调 API 出答案
    2. 没有 → 只输出组装好的 Prompt，你复制到任何网页版都能用

用法：
    python rag_chat.py "西安有哪些 15K 以上的大模型岗"
    python rag_chat.py "需要 PyTorch 的岗位" --no-llm      # 强制只输出 prompt

Prompt 设计的关键：
    - 只允许用资料回答（控制幻觉）
    - 强制标引用编号（可验证）
    - 资料没有就说没有（拒答机制）
"""
import argparse
import json
import os
import time

BASE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(BASE, "chunks.jsonl")
DB_DIR = os.path.join(BASE, "chroma_db")
EMB_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
COLLECTION = "jobs"
RRF_K = 60

SYSTEM_PROMPT = """你是一个严谨的岗位信息助手。请严格依据下面提供的资料回答问题。

【硬性要求】
1. 只使用资料中的信息作答，资料没有明确写出的内容，直接回答「资料中未提及」
2. 每句结论后标注引用编号，如 [1]、[2]
3. 不要编造薪资、公司名、技能要求
4. 如果资料与该问题无关，直接说「资料中没有相关岗位」
5. 回答用简洁的中文，不要重复资料原文，要归纳"""


def load_chunks():
    rows = []
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_context(hits, id2chunk):
    """把检索结果拼成带编号的资料块"""
    parts = []
    for i, (cid, _score) in enumerate(hits, 1):
        c = id2chunk[cid]
        parts.append(f"[{i}] {c['text']}")
    return "\n\n".join(parts)


def build_prompt(query, context):
    return f"""{SYSTEM_PROMPT}

【资料】
{context}

【问题】
{query}

【回答】"""


def call_deepseek(prompt, api_key):
    """调用 DeepSeek API（OpenAI 兼容接口）"""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        return f"[API 错误 {e.code}] {e.read().decode('utf-8', errors='ignore')[:300]}"
    except Exception as e:
        return f"[调用失败] {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--recall", type=int, default=20)
    ap.add_argument("--no-llm", action="store_true", help="只输出 prompt，不调 API")
    ap.add_argument("--save", default=None, help="把 prompt 存到文件")
    ap.add_argument("--no-filter", action="store_true", help="关闭岗位类型过滤")
    args = ap.parse_args()

    raw_query = " ".join(args.query) if args.query else input("查询: ").strip()
    if not raw_query:
        return

    # ---- 查询改写 + 解析薪资约束 ----
    from rag_filter import (
        rewrite_query, filter_by_title, filter_by_salary, extract_salary_floor,
    )
    query = rewrite_query(raw_query)
    salary_floor = extract_salary_floor(raw_query)

    print(f"原始查询: {raw_query}")
    if query != raw_query:
        print(f"改写后:   {query}")
    if salary_floor:
        print(f"薪资下限: {salary_floor:.0f} 元/月")
    print()

    chunks = load_chunks()
    id2chunk = {c["chunk_id"]: c for c in chunks}

    # ---- 检索 + 重排 ----
    print("[1/4] 检索...", end=" ", flush=True)
    t0 = time.time()
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import chromadb
    import jieba
    from rank_bm25 import BM25Okapi

    jieba.setLogLevel(60)
    emb = SentenceTransformer(EMB_MODEL)
    reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    client = chromadb.PersistentClient(path=DB_DIR)
    col = client.get_collection(COLLECTION)

    qv = emb.encode([query], normalize_embeddings=True).tolist()
    r = col.query(query_embeddings=qv, n_results=args.recall, include=["distances"])
    vec = [(x, 1 - d) for x, d in zip(r["ids"][0], r["distances"][0])]

    tokenized = [list(jieba.cut(c["text"])) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    all_ids = [c["chunk_id"] for c in chunks]
    sc = bm25.get_scores(list(jieba.cut(query)))
    bm = sorted(zip(all_ids, sc), key=lambda x: -x[1])[:args.recall]

    fused = {}
    for ranking in (vec, bm):
        for rank, (did, _) in enumerate(ranking, start=1):
            fused[did] = fused.get(did, 0.0) + 1.0 / (RRF_K + rank)
    candidates = [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])][:args.recall]
    print(f"{len(candidates)} 条")

    print("[2/4] 重排...", end=" ", flush=True)
    pairs = [(query, id2chunk[cid]["text"]) for cid in candidates]
    ce = reranker.predict(pairs, show_progress_bar=False)
    top = sorted(zip(candidates, ce), key=lambda x: -x[1])[:args.top]
    print(f"Top {args.top}")

    # ---- 过滤（岗位类型 + 薪资）----
    dropped_info = []
    if not args.no_filter:
        # 先按薪资过滤（在更大范围上过滤，避免过滤后不够）
        top20 = sorted(zip(candidates, ce), key=lambda x: -x[1])[:args.recall]
        if salary_floor:
            top20, dropped_salary = filter_by_salary(top20, id2chunk, salary_floor)
            dropped_info += [f"薪资不符: {t} ({s})" for _, t, s in dropped_salary]
        # 再按岗位类型过滤
        top20, dropped_title = filter_by_title(top20, id2chunk, min_keep=args.top)
        dropped_info += [f"非技术岗: {t}" for _, t in dropped_title]
        top = top20[:args.top]
        if dropped_info:
            print(f"  过滤掉 {len(dropped_info)} 条（薪资/岗位类型不符）")

    # ---- 拼 prompt ----
    print("[3/4] 组装 Prompt...", end=" ", flush=True)
    context = build_context(top, id2chunk)
    prompt = build_prompt(query, context)
    print(f"{len(prompt)} 字符")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"  已存: {args.save}")

    # ---- 生成 ----
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if args.no_llm or not api_key:
        print("[4/4] 无 API key 或 --no-llm：输出 Prompt\n")
        print("=" * 70)
        print(prompt)
        print("=" * 70)
        print("\n提示：把上面这段复制到 DeepSeek / 任何大模型网页版即可。")
        print("       想自动调 API：设置环境变量 DEEPSEEK_API_KEY 后重跑。")
    else:
        print("[4/4] 调用 DeepSeek API...\n")
        answer = call_deepseek(prompt, api_key)
        print("=" * 70)
        print(answer)
        print("=" * 70)

        # 附上引用源的详细信息
        print("\n【引用来源】")
        for i, (cid, score) in enumerate(top, 1):
            c = id2chunk[cid]
            print(f"  [{i}] {c['job_title']} | {c['salary']} | {c['company']}  (rerank={score:.3f})")


if __name__ == "__main__":
    main()
