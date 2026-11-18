"""阶段 3：混合检索（向量 + BM25 + RRF 融合）

目的：补向量检索的短板——对「专有名词、数字、技术栈名」不敏感。

用法：
    python hybrid_search.py "需要 PyTorch 的大模型岗" --top 5
    python hybrid_search.py "RAG" --top 5 --compare   # 对比纯向量 vs 混合

原理：
    1. 向量检索：语义相似（BGE-M3 + Chroma）
    2. BM25 检索：精确词匹配（jieba 分词）
    3. RRF 融合：只看排名不看分值，规避量纲差异
"""
import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(BASE, "chunks.jsonl")
DB_DIR = os.path.join(BASE, "chroma_db")
MODEL_NAME = "BAAI/bge-m3"
COLLECTION = "jobs"

RRF_K = 60  # RRF 平滑常数（论文默认值）


# ---------- 数据加载 ----------
def load_chunks():
    rows = []
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------- 向量检索 ----------
class VectorRetriever:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import chromadb

        self.model = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.col = self.client.get_collection(COLLECTION)

    def search(self, query, top_k=20):
        q_vec = self.model.encode([query], normalize_embeddings=True).tolist()
        res = self.col.query(
            query_embeddings=q_vec,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        sims = [1 - d for d in res["distances"][0]]
        return list(zip(ids, sims))


# ---------- BM25 检索 ----------
class BM25Retriever:
    def __init__(self, chunks):
        import jieba
        from rank_bm25 import BM25Okapi

        jieba.setLogLevel(60)  # 关闭 jieba 的启动日志
        self.chunks = chunks
        self.ids = [c["chunk_id"] for c in chunks]
        # 中文分词：jieba 切成词，再交给 BM25
        self.tokenized = [list(jieba.cut(c["text"])) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)
        self.jieba = jieba

    def search(self, query, top_k=20):
        q_tokens = list(self.jieba.cut(query))
        scores = self.bm25.get_scores(q_tokens)
        # 按分数降序取 Top-K
        ranked = sorted(
            zip(self.ids, scores),
            key=lambda x: -x[1],
        )[:top_k]
        return ranked


# ---------- RRF 融合 ----------
def rrf_fusion(rankings, k=RRF_K):
    """rankings: [[(id, score), ...], [(id, score), ...]]

    RRF 只看排名，不看分值——这是它最大的价值：
    向量相似度是 0-1，BM25 分数可能是 0-20，量纲完全不同，
    加权求和需要反复调参，而 RRF 天然免疫。
    """
    fused = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--top", type=int, default=5, help="最终返回几条")
    ap.add_argument("--recall", type=int, default=20, help="每路召回多少条")
    ap.add_argument("--compare", action="store_true", help="对比纯向量和混合的结果")
    args = ap.parse_args()

    query = " ".join(args.query) if args.query else input("查询: ").strip()
    if not query:
        return

    print(f"查询: {query}")
    print(f"配置: 每路召回 {args.recall} 条 → RRF 融合 → 取 Top {args.top}\n")

    chunks = load_chunks()
    id2chunk = {c["chunk_id"]: c for c in chunks}

    # 向量检索
    print("[1/3] 向量检索...", end=" ", flush=True)
    vr = VectorRetriever()
    vec_hits = vr.search(query, args.recall)
    print(f"{len(vec_hits)} 条")

    # BM25 检索
    print("[2/3] BM25 检索...", end=" ", flush=True)
    bm = BM25Retriever(chunks)
    bm25_hits = bm.search(query, args.recall)
    print(f"{len(bm25_hits)} 条")

    # RRF 融合
    print("[3/3] RRF 融合...", end=" ", flush=True)
    fused = rrf_fusion([vec_hits, bm25_hits])
    print(f"{len(fused)} 条\n")

    if args.compare:
        print("=" * 70)
        print("【纯向量检索】")
        for i, (cid, sim) in enumerate(vec_hits[:args.top], 1):
            c = id2chunk[cid]
            print(f"  {i}. [{sim:.4f}] {c['job_title'][:26]:28} | {c['salary']:12} | {c['company'][:14]}")
        print("\n【纯 BM25 检索】")
        for i, (cid, sc) in enumerate(bm25_hits[:args.top], 1):
            c = id2chunk[cid]
            print(f"  {i}. [{sc:6.2f}] {c['job_title'][:26]:28} | {c['salary']:12} | {c['company'][:14]}")
        print("\n【混合检索（RRF 融合）】")
        for i, (cid, rr) in enumerate(fused[:args.top], 1):
            c = id2chunk[cid]
            in_vec = any(x[0] == cid for x in vec_hits[:args.top])
            in_bm = any(x[0] == cid for x in bm25_hits[:args.top])
            tag = ("向量" if in_vec else "") + ("+BM25" if in_bm else "")
            print(f"  {i}. [{rr:.4f}] {c['job_title'][:26]:28} | {c['salary']:12} | [{tag or '仅融合后'}]")
        print("=" * 70)
    else:
        for i, (cid, rr) in enumerate(fused[:args.top], 1):
            c = id2chunk[cid]
            print(f"[{i}] RRF {rr:.4f} | {c['job_title']}")
            print(f"    公司: {c['company']}   薪资: {c['salary']}")
            print(f"    {c['text'][:140]}...")
            print()


if __name__ == "__main__":
    main()
