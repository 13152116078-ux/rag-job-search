"""阶段 2 验证：检索

用法：
    python search.py "西安有哪些 15K 以上的大模型岗位"
    python search.py "需要 PyTorch 的算法岗" --top 10

输出：Top-K 相关岗位（按相似度降序）
"""
import argparse
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE, "chroma_db")
MODEL_NAME = "BAAI/bge-m3"
COLLECTION = "jobs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="查询语句")
    ap.add_argument("--top", type=int, default=5, help="返回前几条")
    args = ap.parse_args()

    query = " ".join(args.query) if args.query else input("查询: ").strip()
    if not query:
        print("查询不能为空")
        return

    if not os.path.isdir(DB_DIR):
        print(f"向量库不存在：{DB_DIR}")
        print("先跑 python build-index.py")
        return

    print(f"查询: {query}\n")

    # ---- 编码查询 ----
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    # BGE 系列建议给 query 加指令前缀（中文任务）
    q_vec = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    # ---- 检索 ----
    import chromadb
    client = chromadb.PersistentClient(path=DB_DIR)
    col = client.get_collection(COLLECTION)

    res = col.query(
        query_embeddings=q_vec,
        n_results=args.top,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    print("=" * 70)
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        # cosine 距离 → 相似度
        sim = 1 - dist
        print(f"\n[{i}] 相似度 {sim:.4f}  |  {meta.get('job_title', '')}")
        print(f"    公司: {meta.get('company', '')}   薪资: {meta.get('salary', '')}")
        print(f"    {doc[:160]}...")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
