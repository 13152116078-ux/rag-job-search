"""简历匹配：把简历和岗位库做匹配打分

思路：
1. 从简历里抽技能（对照技能词表 + 简单规则）
2. 用简历精华当 query 去检索（复用 RAG 检索链路）
3. 对每个候选岗位算三部分分数：
   - 语义分：检索得分（简历整体 vs 岗位描述）
   - 技能分：技能重合度（Jaccard 加权）
   - 硬性分：经验/学历是否满足
4. 生成匹配理由（命中的技能 + 缺的技能）

用法：
    from resume_match import match_resume
    out = match_resume("简历全文...", top=10)
"""
import re

# ---------- 技能词表（覆盖 AI/LLM 岗位常见要求） ----------
SKILL_GROUPS = {
    "编程语言": ["python", "java", "c++", "c", "go", "javascript", "typescript", "sql", "shell", "scala", "rust"],
    "AI基础": ["机器学习", "深度学习", "神经网络", "transformer", "cnn", "rnn", "nlp", "计算机视觉", "cv", "多模态", "强化学习"],
    "大模型": ["大模型", "llm", "gpt", "bert", "llama", "qwen", "微调", "lora", "qlora", "sft", "rlhf", "prompt", "rag", "agent",
               "langchain", "llamaindex", "向量数据库", "向量检索", "embedding", "rerank", "vllm", "推理优化", "量化"],
    "框架工具": ["pytorch", "tensorflow", "keras", "sklearn", "scikit-learn", "flask", "fastapi", "django", "spring", "vue", "react",
                 "docker", "kubernetes", "k8s", "git", "linux", "mysql", "redis", "mongodb", "kafka", "spark", "hadoop"],
    "视觉": ["yolo", "opencv", "图像处理", "目标检测", "分割", "ocr", "人脸识别"],
    "数据": ["数据标注", "数据清洗", "数据分析", "pandas", "numpy", "爬虫", "etl"],
}

ALL_SKILLS = []
for group, items in SKILL_GROUPS.items():
    for s in items:
        ALL_SKILLS.append((s.lower(), group))

# 别名词表：一个概念可能有多种写法
ALIASES = {
    "大模型": ["大模型", "大语言模型", "llm", "基础模型"],
    "机器学习": ["机器学习", "ml", "machine learning"],
    "深度学习": ["深度学习", "dl", "deep learning"],
    "计算机视觉": ["计算机视觉", "cv", "视觉算法", "图像算法"],
    "自然语言处理": ["自然语言处理", "nlp"],
    "检索增强": ["rag", "检索增强", "知识库问答"],
    "微调": ["微调", "fine-tune", "finetune", "finetuning", "lora", "qlora"],
}

# ---------- 学历等级 ----------
DEGREE_LEVEL = {
    "学历不限": 0, "不限": 0, "初中": 1, "中专/中技": 2,
    "高中": 3, "大专": 4, "本科": 5, "硕士": 6, "博士": 7,
}


def normalize(text: str) -> str:
    return (text or "").lower()


def extract_skills(text: str):
    """从文本里抽取技能，返回 {技能名: 所属类别}"""
    t = normalize(text)
    found = {}
    # 直接匹配
    for skill, group in ALL_SKILLS:
        # 英文用单词边界，中文直接包含
        if re.search(r"[a-z]", skill):
            if re.search(r"(?<![a-z])" + re.escape(skill) + r"(?![a-z])", t):
                found[skill] = group
        else:
            if skill in t:
                found[skill] = group
    # 别名兜底
    for canonical, alias_list in ALIASES.items():
        for a in alias_list:
            if a in t:
                found[canonical] = found.get(canonical, _guess_group(canonical))
                break
    return found


def _guess_group(skill):
    for group, items in SKILL_GROUPS.items():
        if any(skill in i or i in skill for i in items):
            return group
    return "其他"


def parse_degree_req(degree_str: str) -> int:
    """把岗位学历要求转成档位"""
    s = (degree_str or "").strip()
    for k, v in DEGREE_LEVEL.items():
        if k in s:
            return v
    return 5  # 默认按本科


def parse_exp_req(exp_str: str):
    """解析经验要求，返回 (最低年, 最高年)；None 表示不限"""
    s = (exp_str or "").strip()
    if not s or "不限" in s or "应届" in s or "在校" in s:
        return (0, 99)
    m = re.search(r"(\d+)-(\d+)", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(\d+)\s*年", s)
    if m:
        return (int(m.group(1)), 99)
    return (0, 99)


def parse_resume_edu(text: str):
    """从简历里猜学历（取最高）"""
    t = normalize(text)
    level = 0
    if "博士" in t or "phd" in t:
        level = 7
    elif "硕士" in t or "研究生" in t or "master" in t:
        level = 6
    elif "本科" in t or "学士" in t or "bachelor" in t:
        level = 5
    elif "大专" in t or "专科" in t:
        level = 4
    return level


def parse_resume_years(text: str) -> int:
    """从简历里猜工作年数（实习也算，按 0.5 折算）"""
    t = normalize(text)
    if "应届" in t or "在校" in t or "大三" in t or "大四" in t:
        return 0
    m = re.search(r"(\d+)\s*年.*(经验|工作)", t)
    if m:
        return int(m.group(1))
    return 0


def match_score(resume_text: str, job: dict):
    """给单个岗位打分，返回 (总分, 明细)"""
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(
        (job.get("title", "") + " " + job.get("skills", "") + " " + job.get("text", ""))
    )

    # --- 技能分 ---
    if not job_skills:
        skill_score = 0.5  # 岗位没写技能要求，给中性分
        hit_skills, miss_skills = [], []
    else:
        hit = set(resume_skills) & set(job_skills)
        miss = set(job_skills) - set(resume_skills)
        n = len(job_skills)
        skill_score = len(hit) / n
        # 惩罚：技能要求太少的岗位（只要 1-2 个技能），命中率高不可信
        if n < 3:
            skill_score *= (n / 3.0)
        hit_skills = sorted(hit)
        miss_skills = sorted(miss)

    # --- 硬性条件 ---
    edu_ok = True
    edu_msg = ""
    req_edu = parse_degree_req(job.get("degree", ""))
    my_edu = parse_resume_edu(resume_text)
    if my_edu and req_edu > my_edu:
        edu_ok = False
        edu_msg = "学历不满足"

    exp_ok = True
    exp_msg = ""
    exp_min, exp_max = parse_exp_req(job.get("experience", ""))
    my_years = parse_resume_years(resume_text)
    if my_years < exp_min:
        # 差 1 年内不扣，差距大才降档
        gap = exp_min - my_years
        if gap >= 2:
            exp_ok = False
            exp_msg = f"经验差 {gap} 年"

    # --- 总分 ---
    score = skill_score * 0.7 + (0.3 if edu_ok and exp_ok else 0.0)
    if not edu_ok or not exp_ok:
        score *= 0.6

    return {
        "score": round(score * 100, 1),
        "skill_match": round(skill_score * 100, 1),
        "hit_skills": hit_skills[:8],
        "miss_skills": miss_skills[:6],
        "edu_ok": edu_ok,
        "exp_ok": exp_ok,
        "note": " ".join([x for x in [edu_msg, exp_msg] if x]),
    }


def build_resume_query(resume_text: str, max_len: int = 400) -> str:
    """把简历压缩成一段检索用的 query（取技能和经历的精华）"""
    skills = list(extract_skills(resume_text).keys())
    # 取简历前 max_len 字
    head = re.sub(r"\s+", " ", resume_text)[:max_len]
    skill_line = " ".join(skills[:60])
    return (head + " " + skill_line).strip()
