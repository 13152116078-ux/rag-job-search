"""评测 2：加入重排层，对比三档

    A. 纯向量
    B. 混合 RRF
    C. 混合 RRF + CrossEncoder 重排

用法：python eval_rerank.py --sample 20
"""
import argparse
import json
import os
import random
import time

BASE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(BASE, "chunks.jsonl")
DB_DIR = os.path.join(BASE, "chroma_db")
EMB_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
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
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--recall", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    chunks = load_chunks()
    id2chunk = {c["chunk_id"]: c for c in chunks}
    pool = [c for c in chunks if c["job_title"].strip()]
    sample = random.sample(pool, min(args.sample, len(pool)))
    queries = [(c["chunk_id"], c["job_title"]) for c in sample]

    print(f"评测集: {len(queries)} 条 | 指标 Recall@{args.topk} | 召回 {args.recall} 条\n")

    from sentence_transformers import SentenceTransformer, CrossEncoder
    import chromadb
    import jieba
    from rank_bm25 import BM25Okapi

    jieba.setLogLevel(60)
    print("[加载] 模型...", end=" ", flush=True)
    t0 = time.time()
    emb = SentenceTransformer(EMB_MODEL)
    reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    print(f"{time.time() - t0:.1f}s")

    client = chromadb.PersistentClient(path=DB_DIR)
    col = client.get_collection(COLLECTION)

    print("[构建] BM25...", end=" ", flush=True)
    tokenized = [list(jieba.cut(c["text"])) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    all_ids = [c["chunk_id"] for c in chunks]
    print("ok\n")

    hits = {"vector": 0, "hybrid": 0, "rerank": 0}
    t_start = time.time()

    for i, (gold_id, q) in enumerate(queries, 1):
        # A. 纯向量
        qv = emb.encode([q], normalize_embeddings=True).tolist()
        r = col.query(query_embeddings=qv, n_results=args.recall, include=["distances"])
        vec = [(x, 1 - d) for x, d in zip(r["ids"][0], r["distances"][0])]

        # B. 混合
        sc = bm25.get_scores(list(jieba.cut(q)))
        bm = sorted(zip(all_ids, sc), key=lambda x: -x[1])[:args.recall]
        fused = {}
        for ranking in (vec, bm):
            for rank, (did, _) in enumerate(ranking, start=1):
                fused[did] = fused.get(did, 0.0) + 1.0 / (RRF_K + rank)
        hybrid = [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])][:args.recall]

        # C. 重排（只对混合的前 20 条重排）
        pairs = [(q, id2chunk[cid]["text"]) for cid in hybrid]
        ce = reranker.predict(pairs, show_progress_bar=False)
        reranked = [cid for cid, _ in sorted(zip(hybrid, ce), key=lambda x: -x[1])]

        if any(x[0] == gold_id for x in vec[:args.topk]):
            hits["vector"] += 1
        if gold_id in hybrid[:args.topk]:
            hits["hybrid"] += 1
        if gold_id in reranked[:args.topk]:
            hits["rerank"] += 1

        if i % 5 == 0:
            print(f"  进度 {i}/{len(queries)}  ({time.time() - t_start:.0f}s)", flush=True)

    n = len(queries)
    print("\n" + "=" * 64)
    print(f"Recall@{args.topk}  （样本 {n}，总用时 {time.time() - t_start:.1f}s）")
    print("=" * 64)
    names = [("纯向量", "vector"), ("混合 RRF", "hybrid"), ("混合+重排", "rerank")]
    for label, key in names:
        c = hits[key]
        print(f"  {label:12} {c:3}/{n}  =  {c / n * 100:5.1f}%")

    v = hits["vector"] / n * 100
    h = hits["hybrid"] / n * 100
    r = hits["rerank"] / n * 100
    print("\n" + "-" * 64)
    print(f"  混合 vs 纯向量:      {h - v:+.1f} 个点")
    print(f"  +重排 vs 混合:       {r - h:+.1f} 个点")
    print(f"  +重排 vs 纯向量:     {r - v:+.1f} 个点")
    print("=" * 64)

    out = os.path.join(BASE, "eval_rerank_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "sample": n, "topk": args.topk,
            "vector": v, "hybrid": h, "rerank": r,
            "delta_hybrid": h - v, "delta_rerank": r - h, "delta_total": r - v,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已存: {out}")


if __name__ == "__main__":
    main()
