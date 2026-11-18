"""阶段 4：Cross-encoder 重排

架构：
    混合检索（向量+BM25+RRF）召回 Top-20
        ↓
    CrossEncoder (bge-reranker-v2-m3) 精排
        ↓
    Top-5

为什么两级：
    Bi-encoder 把 query 和 doc 分别编码，快但两者之间无交互。
    Cross-encoder 把 (query, doc) 拼起来喂进模型，注意力在两者间充分交互，准但慢。
    所以用便宜的粗召回把候选缩到 20 条，再用贵的模型精挑 5 条。

用法：
    python rerank_search.py "需要 PyTorch 的大模型岗" --top 5
    python rerank_search.py "RAG" --top 5 --compare
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
    ap.add_argument("query", nargs="*")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--recall", type=int, default=20, help="粗召回到多少条")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    query = " ".join(args.query) if args.query else input("查询: ").strip()
    if not query:
        return

    print(f"查询: {query}")
    print(f"流程: 混合召回 {args.recall} 条 → CrossEncoder 重排 → Top {args.top}\n")

    chunks = load_chunks()
    id2chunk = {c["chunk_id"]: c for c in chunks}

    # ---- 粗召回：向量 + BM25 + RRF ----
    print("[1/4] 加载模型...", end=" ", flush=True)
    t0 = time.time()
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import chromadb
    import jieba
    from rank_bm25 import BM25Okapi

    jieba.setLogLevel(60)
    emb = SentenceTransformer(EMB_MODEL)
    reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    print(f"{time.time() - t0:.1f}s")

    client = chromadb.PersistentClient(path=DB_DIR)
    col = client.get_collection(COLLECTION)

    print("[2/4] 混合召回...", end=" ", flush=True)
    t1 = time.time()
    qv = emb.encode([query], normalize_embeddings=True).tolist()
    r = col.query(query_embeddings=qv, n_results=args.recall, include=["distances"])
    vec_hits = [(i, 1 - d) for i, d in zip(r["ids"][0], r["distances"][0])]

    tokenized = [list(jieba.cut(c["text"])) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    all_ids = [c["chunk_id"] for c in chunks]
    scores = bm25.get_scores(list(jieba.cut(query)))
    bm_hits = sorted(zip(all_ids, scores), key=lambda x: -x[1])[:args.recall]

    fused = {}
    for ranking in (vec_hits, bm_hits):
        for rank, (did, _) in enumerate(ranking, start=1):
            fused[did] = fused.get(did, 0.0) + 1.0 / (RRF_K + rank)
    candidates = [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])][:args.recall]
    print(f"{len(candidates)} 条，{time.time() - t1:.1f}s")

    # ---- 精排 ----
    print("[3/4] CrossEncoder 重排...", end=" ", flush=True)
    t2 = time.time()
    pairs = [(query, id2chunk[cid]["text"]) for cid in candidates]
    ce_scores = reranker.predict(pairs, show_progress_bar=False)
    reranked = sorted(zip(candidates, ce_scores), key=lambda x: -x[1])
    print(f"{time.time() - t2:.1f}s")

    # ---- 输出 ----
    print("[4/4] 完成\n")

    if args.compare:
        print("=" * 74)
        print("【重排前（RRF 排序）】Top 5")
        for i, cid in enumerate(candidates[:args.top], 1):
            c = id2chunk[cid]
            print(f"  {i}. {c['job_title'][:28]:30} | {c['salary']:12} | {c['company'][:12]}")
        print("\n【重排后（CrossEncoder 排序）】Top 5")
        for i, (cid, sc) in enumerate(reranked[:args.top], 1):
            c = id2chunk[cid]
            print(f"  {i}. [{sc:6.3f}] {c['job_title'][:26]:28} | {c['salary']:12} | {c['company'][:12]}")
        print("=" * 74)
    else:
        for i, (cid, sc) in enumerate(reranked[:args.top], 1):
            c = id2chunk[cid]
            print(f"[{i}] 重排得分 {sc:.4f} | {c['job_title']}")
            print(f"    公司: {c['company']}   薪资: {c['salary']}")
            print(f"    {c['text'][:140]}...")
            print()


if __name__ == "__main__":
    main()
