"""阶段 2：向量化 + 入库

流程：chunks.jsonl → BGE-M3 编码 → 存进 Chroma

用法：
    python build-index.py

产物：./chroma_db/ 目录（持久化向量库）
"""
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(BASE, "chunks.jsonl")
DB_DIR = os.path.join(BASE, "chroma_db")

MODEL_NAME = "BAAI/bge-m3"
COLLECTION = "jobs"


def load_chunks():
    rows = []
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    print("=" * 60)
    print("阶段 2：向量化 + 入库")
    print("=" * 60)

    # ---- 1. 读 chunks ----
    chunks = load_chunks()
    print(f"\n[1/4] 读取 chunks: {len(chunks)} 条")

    # ---- 2. 加载模型 ----
    print(f"\n[2/4] 加载嵌入模型 {MODEL_NAME}（首次会下载约 2GB）...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print(f"   模型加载完成，用时 {time.time() - t0:.1f}s")
    print(f"   向量维度: {model.get_sentence_embedding_dimension()}")

    # ---- 3. 编码 ----
    print(f"\n[3/4] 编码 {len(chunks)} 条文本...", flush=True)
    t0 = time.time()
    texts = [c["text"] for c in chunks]
    # 注意：不要开 show_progress_bar，tqdm 输出到管道会阻塞
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=False,
    ).tolist()
    print(f"   编码完成，用时 {time.time() - t0:.1f}s", flush=True)
    print(f"   向量形状: {len(embeddings)} x {len(embeddings[0])}", flush=True)

    # ---- 4. 写入 Chroma ----
    print(f"\n[4/4] 写入 Chroma（{DB_DIR}）...")
    import chromadb
    client = chromadb.PersistentClient(path=DB_DIR)

    # 重建集合，避免重复写入
    try:
        client.delete_collection(COLLECTION)
        print("   已删除旧集合")
    except Exception:
        pass

    col = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # 分批写入，避免一次太大
    BATCH = 100
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        col.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings[i:i + BATCH],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "job_id": c["job_id"],
                    "job_title": c["job_title"],
                    "company": c["company"],
                    "salary": c["salary"],
                }
                for c in batch
            ],
        )
        print(f"   写入 {min(i + BATCH, len(chunks))}/{len(chunks)}")

    print(f"\n完成。集合 {COLLECTION} 共 {col.count()} 条向量")
    print("下一步：python search.py \"你的问题\"")


if __name__ == "__main__":
    main()
