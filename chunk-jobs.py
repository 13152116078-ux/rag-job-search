"""把 jobs.jsonl 切成 RAG 用的 chunks.jsonl

策略：
- 每条岗位先拼成一段自然语言描述（标题/薪资/公司/经验/学历/技能/福利）
- 按字符数切块，chunk_size=400，overlap=80
- 保留 job_id / title / company 等元数据，方便溯源引用

用法：
    python chunk-jobs.py
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "jobs.jsonl")
OUT = os.path.join(BASE, "chunks.jsonl")

CHUNK_SIZE = 400
OVERLAP = 80


def build_document(job):
    """把一条岗位拼成自然语言段落。"""
    parts = []
    parts.append(f"岗位名称：{job.get('title', '')}")
    if job.get("salary"):
        parts.append(f"薪资：{job['salary']}")
    if job.get("company"):
        parts.append(f"公司：{job['company']}")
    if job.get("industry"):
        parts.append(f"行业：{job['industry']}")
    if job.get("scale"):
        parts.append(f"公司规模：{job['scale']}")
    if job.get("stage"):
        parts.append(f"融资阶段：{job['stage']}")
    loc = "".join([job.get("city", ""), job.get("area", "")])
    if loc:
        parts.append(f"工作地点：{loc}")
    exp = job.get("experience", "")
    deg = job.get("degree", "")
    if exp or deg:
        parts.append(f"经验与学历要求：{exp or '不限'}，{deg or '不限'}")
    if job.get("skills"):
        parts.append(f"技能要求：{job['skills']}")
    if job.get("welfare"):
        parts.append(f"福利待遇：{job['welfare']}")
    return "。".join(parts) + "。"


def split_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """按字符切块，带重叠。"""
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def main():
    jobs = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))

    out = []
    idx = 0
    for job in jobs:
        doc = build_document(job)
        pieces = split_text(doc)
        for i, piece in enumerate(pieces):
            idx += 1
            out.append({
                "chunk_id": f"c{idx:05d}",
                "job_id": job.get("id", ""),
                "job_title": job.get("title", ""),
                "company": job.get("company", ""),
                "salary": job.get("salary", ""),
                "chunk_index": i,
                "chunk_total": len(pieces),
                "text": piece,
            })

    with open(OUT, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"文档数: {len(jobs)}")
    print(f"chunk 数: {len(out)}")
    print(f"平均每文档: {len(out) / max(len(jobs), 1):.2f} 块")
    lens = [len(c["text"]) for c in out]
    print(f"chunk 长度: min={min(lens)} max={max(lens)} avg={sum(lens)//len(lens)}")
    print(f"输出: {OUT} ({os.path.getsize(OUT)} B)")


if __name__ == "__main__":
    main()
