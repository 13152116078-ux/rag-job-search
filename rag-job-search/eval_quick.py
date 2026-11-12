"""阶段 4（快速版）：自动评测 —— 纯向量 vs 混合检索

做法：用每条岗位的「标题」当查询，期望召回它自己。
      这是检索自检（self-retrieval），不是严格的人工评测，
      但能在几分钟内看出两路方案的相对差异。

指标：Recall@K（目标 chunk 是否落在 Top-K）

用法：
    python eval_quick.py
    python eval_quick.py --sample 50 --topk 5
"""
import argparse
import json
import os
import random
import time

BASE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(BASE, "chunks.jsonl")
DB_DIR = os.path.join(BASE, "chroma_db")
MODEL_NAME = "BAAI/bge-m3"
COLLECTION = "jobs"
RRF_K = 60


def load_chunks():
    rows = []
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50, help="采样多少个 query")
    ap.add_argument("--topk", type=int, default=5, help="评估 Recall@K")
    ap.add_argument("--recall", type=int, default=20, help="每路召回条数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    chunks = load_chunks()
    id2chunk = {c["chunk_id"]: c for c in chunks}

    # ---- 构造评测集：用标题当查询 ----
    # 过滤掉空标题
    pool = [c for c in chunks if c["job_title"].strip()]
    sample = random.sample(pool, min(args.sample, len(pool)))

    queries = [(c["chunk_id"], c["job_title"]) for c in sample]
    print(f"评测集: {len(queries)} 条 query（来自岗位标题）")
    print(f"指标: Recall@{args.topk}\n")

    # ---- 加载检索器 ----
    print("[加载] 模型 + 向量库...", flush=True)
    from sentence_transformers import SentenceTransformer
    import chromadb
    import jieba
    from rank_bm25 import BM25Okapi

    jieba.setLogLevel(60)
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_DIR)
    col = client.get_collection(COLLECTION)

    print("[构建] BM25 索引...", flush=True)
    tokenized = [list(jieba.cut(c["text"])) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    all_ids = [c["chunk_id"] for c in chunks]

    def vec_search(q, k):
        qv = model.encode([q], normalize_embeddings=True).tolist()
        r = col.query(query_embeddings=qv, n_results=k, include=["distances"])
        return [(i, 1 - d) for i, d in zip(r["ids"][0], r["distances"][0])]

    def bm_search(q, k):
        scores = bm25.get_scores(list(jieba.cut(q)))
        ranked = sorted(zip(all_ids, scores), key=lambda x: -x[1])[:k]
        return ranked

    def rrf(rankings, k=RRF_K):
        fused = {}
        for ranking in rankings:
            for rank, (did, _) in enumerate(ranking, start=1):
                fused[did] = fused.get(did, 0.0) + 1.0 / (k + rank)
        return sorted(fused.items(), key=lambda x: -x[1])

    # ---- 逐条评测 ----
    print("[搜索] 开始评测...\n", flush=True)
    hits = {"vector": 0, "bm25": 0, "hybrid": 0}
    t0 = time.time()

    for i, (gold_id, q) in enumerate(queries, 1):
        v = vec_search(q, args.recall)
        b = bm_search(q, args.recall)
        h = rrf([v, b])

        if any(x[0] == gold_id for x in v[:args.topk]):
            hits["vector"] += 1
        if any(x[0] == gold_id for x in b[:args.topk]):
            hits["bm25"] += 1
        if any(x[0] == gold_id for x in h[:args.topk]):
            hits["hybrid"] += 1

        if i % 10 == 0:
            print(f"  进度 {i}/{len(queries)}", flush=True)

    n = len(queries)
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"Recall@{args.topk}  （样本 {n} 条，用时 {elapsed:.1f}s）")
    print("=" * 60)
    for name, cnt in [("纯向量", hits["vector"]), ("纯 BM25", hits["bm25"]), ("混合 RRF", hits["hybrid"])]:
        print(f"  {name:10} {cnt:3}/{n}  =  {cnt / n * 100:5.1f}%")

    v_pct = hits["vector"] / n * 100
    h_pct = hits["hybrid"] / n * 100
    b_pct = hits["bm25"] / n * 100
    print("\n" + "-" * 60)
    print(f"  混合 vs 纯向量: {h_pct - v_pct:+.1f} 个点")
    print(f"  混合 vs 纯BM25: {h_pct - b_pct:+.1f} 个点")
    print("=" * 60)

    # 保存结果
    out = os.path.join(BASE, "eval_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "sample": n,
            "topk": args.topk,
            "recall": {"vector": hits["vector"] / n, "bm25": hits["bm25"] / n, "hybrid": hits["hybrid"] / n},
            "delta_hybrid_vs_vector": h_pct - v_pct,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已存: {out}")


if __name__ == "__main__":
    main()
