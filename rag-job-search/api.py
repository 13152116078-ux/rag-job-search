"""阶段 7：FastAPI 服务化

把整个 RAG 链路包成 HTTP 接口。

启动：
    uvicorn api:app --host 127.0.0.1 --port 8000

接口：
    GET  /health              健康检查
    POST /search              只检索，返回岗位列表
    POST /chat                检索 + 拼 prompt（无 key 返回 prompt）
    GET  /stats               库统计

示例：
    curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" -d '{"query":"15K以上大模型岗"}'
"""
import json
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(BASE, "chunks.jsonl")
DB_DIR = os.path.join(BASE, "chroma_db")
EMB_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
COLLECTION = "jobs"
RRF_K = 60

app = FastAPI(title="RAG 岗位检索 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------- 中文界面（首页） ----------
from fastapi.responses import FileResponse

_WEB_INDEX = os.path.join(BASE, "web", "index.html")

@app.get("/", include_in_schema=False)
def home():
    if os.path.exists(_WEB_INDEX):
        return FileResponse(_WEB_INDEX, media_type="text/html")
    return {"msg": "中文页面缺失，请访问 /docs"}



# ---------- 请求模型 ----------
class SearchReq(BaseModel):
    query: str
    top: int = 5
    recall: int = 20
    filter: bool = True


class ChatReq(BaseModel):
    query: str
    top: int = 5


# ---------- 懒加载（第一次请求才加载模型，避免启动太慢） ----------
_state = {}


def _load():
    if _state.get("ready"):
        return _state

    t0 = time.time()
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import chromadb
    import jieba
    from rank_bm25 import BM25Okapi

    jieba.setLogLevel(60)
    chunks = []
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    _state["id2chunk"] = {c["chunk_id"]: c for c in chunks}
    _state["emb"] = SentenceTransformer(EMB_MODEL)
    _state["reranker"] = CrossEncoder(RERANK_MODEL, max_length=512)
    client = chromadb.PersistentClient(path=DB_DIR)
    _state["col"] = client.get_collection(COLLECTION)
    _state["bm25"] = BM25Okapi([list(jieba.cut(c["text"])) for c in chunks])
    _state["all_ids"] = [c["chunk_id"] for c in chunks]
    _state["jieba"] = jieba
    _state["chunks"] = chunks
    _state["ready"] = True
    _state["load_seconds"] = round(time.time() - t0, 1)
    return _state


# ---------- 核心检索 ----------
def _search(query: str, top: int, recall: int, do_filter: bool):
    from rag_filter import (
        rewrite_query, filter_by_title, filter_by_salary, extract_salary_floor,
    )
    st = _load()
    raw_query = query
    q = rewrite_query(query)
    floor = extract_salary_floor(query)

    # 向量
    qv = st["emb"].encode([q], normalize_embeddings=True).tolist()
    r = st["col"].query(query_embeddings=qv, n_results=recall, include=["distances"])
    vec = [(i, 1 - d) for i, d in zip(r["ids"][0], r["distances"][0])]

    # BM25
    sc = st["bm25"].get_scores(list(st["jieba"].cut(q)))
    bm = sorted(zip(st["all_ids"], sc), key=lambda x: -x[1])[:recall]

    # RRF
    fused = {}
    for ranking in (vec, bm):
        for rank, (did, _) in enumerate(ranking, start=1):
            fused[did] = fused.get(did, 0.0) + 1.0 / (RRF_K + rank)
    candidates = [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])][:recall]

    # 重排
    pairs = [(q, st["id2chunk"][c]["text"]) for c in candidates]
    ce = st["reranker"].predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidates, ce), key=lambda x: -x[1])

    dropped = []
    if do_filter:
        if floor:
            ranked, ds = filter_by_salary(ranked, st["id2chunk"], floor)
            dropped += [f"薪资不符: {t}({s})" for _, t, s in ds]
        ranked, dt = filter_by_title(ranked, st["id2chunk"], min_keep=top)
        dropped += [f"非技术岗: {t}" for _, t in dt]

    results = []
    for cid, score in ranked[:top]:
        c = st["id2chunk"][cid]
        results.append({
            "chunk_id": cid,
            "title": c.get("job_title", ""),
            "company": c.get("company", ""),
            "salary": c.get("salary", ""),
            # 注意：chunks.jsonl 里没有这两项，用 get 兜底
            "experience": c.get("experience", ""),
            "degree": c.get("degree", ""),
            "skills": c.get("skills", ""),
            "score": round(float(score), 4),
        })

    return {
        "raw_query": raw_query,
        "query": q,
        "salary_floor": floor,
        "results": results,
        "dropped": dropped,
        "count": len(results),
    }


# ---------- 接口 ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": bool(_state.get("ready")),
        "load_seconds": _state.get("load_seconds"),
        "chunks": len(_state.get("chunks", [])) or None,
    }


@app.get("/stats")
def stats():
    st = _load()
    return {
        "chunks": len(st["chunks"]),
        "collection": COLLECTION,
        "emb_model": EMB_MODEL,
        "rerank_model": RERANK_MODEL,
        "dim": 1024,
    }


@app.post("/search")
def search(req: SearchReq):
    t0 = time.time()
    try:
        out = _search(req.query, req.top, req.recall, req.filter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    out["elapsed_ms"] = round((time.time() - t0) * 1000)
    return out



# ---------- 简历匹配 ----------
class MatchReq(BaseModel):
    resume: str
    top: int = 10
    city_filter: str = ""


@app.post("/match")
def match_resume_endpoint(req: MatchReq):
    """上传简历文本，返回匹配的岗位列表"""
    from resume_match import match_score, build_resume_query
    t0 = time.time()

    if not req.resume.strip():
        raise HTTPException(status_code=400, detail="简历内容不能为空")

    # 用简历精华当 query 检索
    q = build_resume_query(req.resume)
    try:
        out = _search(q, req.top * 3, 30, False)  # 先多召回一些，后面再精排
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    st = _load()
    scored = []
    for r in out["results"]:
        cid = r["chunk_id"]
        job = dict(st["id2chunk"][cid])
        m = match_score(req.resume, job)

        # 城市过滤
        if req.city_filter and req.city_filter not in job.get("city", ""):
            continue

        scored.append({
            "title": job.get("job_title", ""),
            "company": job.get("company", ""),
            "salary": job.get("salary", ""),
            "city": job.get("city", ""),
            "experience": job.get("experience", ""),
            "degree": job.get("degree", ""),
            "skills": job.get("skills", ""),
            "match_score": m["score"],
            "skill_match": m["skill_match"],
            "hit_skills": m["hit_skills"],
            "miss_skills": m["miss_skills"],
            "note": m["note"],
        })

    scored.sort(key=lambda x: -x["match_score"])

    return {
        "count": len(scored),
        "results": scored[: req.top],
        "elapsed_ms": round((time.time() - t0) * 1000),
    }
@app.post("/chat")
def chat(req: ChatReq):
    """返回拼好的 prompt（有 DEEPSEEK_API_KEY 时顺便出答案）"""
    t0 = time.time()
    try:
        out = _search(req.query, req.top, 20, True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    st = _load()
    ctx = "\n\n".join(
        f"[{i}] {st['id2chunk'][r['chunk_id']]['text']}"
        for i, r in enumerate(out["results"], 1)
    )
    prompt = (
        "你是一个严谨的岗位信息助手。请严格依据下面提供的资料回答问题。\n\n"
        "【硬性要求】\n"
        "1. 只使用资料中的信息作答，资料没有明确写出的内容，直接回答「资料中未提及」\n"
        "2. 每句结论后标注引用编号，如 [1]、[2]\n"
        "3. 不要编造薪资、公司名、技能要求\n"
        "4. 如果资料与该问题无关，直接说「资料中没有相关岗位」\n"
        "5. 回答用简洁的中文，不要重复资料原文，要归纳\n\n"
        f"【资料】\n{ctx}\n\n【问题】\n{req.query}\n\n【回答】"
    )

    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    answer = None
    if key:
        try:
            import urllib.request
            payload = json.dumps({
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }).encode()
            r = urllib.request.Request(
                "https://api.deepseek.com/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                method="POST",
            )
            with urllib.request.urlopen(r, timeout=60) as resp:
                answer = json.loads(resp.read())["choices"][0]["message"]["content"]
        except Exception as e:
            answer = f"[API 调用失败] {e}"

    return {
        "query": req.query,
        "sources": out["results"],
        "dropped": out["dropped"],
        "prompt": prompt,
        "answer": answer,
        "need_api_key": answer is None,
        "elapsed_ms": round((time.time() - t0) * 1000),
    }
