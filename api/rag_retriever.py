"""
RAG 检索模块：从 knowledge_base.json 中检索与用户输入相关的命名策略、行业案例和文化素材。
"""
import json
import os
from typing import Optional

_kb_cache: Optional[dict] = None

def _load_knowledge_base() -> dict:
    global _kb_cache
    if _kb_cache is None:
        kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
        with open(kb_path, "r", encoding="utf-8") as f:
            _kb_cache = json.load(f)
    return _kb_cache


def _match_industry(industry: str, kb: dict) -> Optional[dict]:
    """模糊匹配行业，返回最相关的行业案例"""
    industry_cases = kb.get("industry_cases", {})
    
    # 精确匹配
    if industry in industry_cases:
        return industry_cases[industry]
    
    # 模糊匹配：行业名包含在 key 中或 key 包含在行业名中
    for key, cases in industry_cases.items():
        if key in industry or industry in key:
            return cases
    
    # 关键词匹配
    industry_keywords = {
        "新能源汽车": ["汽车", "车", "出行", "新能源", "电车", "电动"],
        "科技互联网": ["科技", "互联网", "软件", "IT", "技术", "SaaS", "人工智能", "AI"],
        "食品饮料": ["食品", "饮料", "零食", "饮品", "茶", "咖啡", "酒", "餐饮"],
        "母婴": ["母婴", "婴儿", "儿童", "宝宝", "孕", "童装"],
        "美妆护肤": ["美妆", "护肤", "化妆", "美容", "彩妆", "面膜"],
        "教育培训": ["教育", "培训", "学习", "课程", "在线教育", "考试"],
        "医疗健康": ["医疗", "健康", "医药", "保健", "养生", "医院"],
        "金融理财": ["金融", "银行", "投资", "保险", "理财", "支付"],
    }
    
    for key, keywords in industry_keywords.items():
        if any(kw in industry for kw in keywords):
            return industry_cases.get(key)
    
    return None


def retrieve_context(industry: str, keywords: str) -> dict:
    """
    根据用户输入的行业和关键词，从知识库中检索相关上下文。
    
    返回:
        {
            "strategies": 所有命名策略列表,
            "industry_cases": 匹配到的行业案例(可能为None),
            "cultural_refs": 文化素材,
            "evaluation_dimensions": 评分维度,
            "few_shot_prompt": 构造好的 few-shot 示例文本
        }
    """
    kb = _load_knowledge_base()
    
    strategies = kb.get("naming_strategies", [])
    industry_cases = _match_industry(industry, kb)
    cultural_refs = kb.get("cultural_references", {})
    eval_dims = kb.get("evaluation_dimensions", {})
    
    # 构造 few-shot 示例文本
    few_shot_lines = []
    if industry_cases:
        few_shot_lines.append(f"【{industry} 行业成功案例参考】")
        for case in industry_cases.get("successful_names", [])[:3]:
            few_shot_lines.append(f"  · {case['name']}（策略：{case['strategy']}）—— {case['reason']}")
        if industry_cases.get("common_elements"):
            few_shot_lines.append(f"  常用元素字：{'、'.join(industry_cases['common_elements'])}")
        if industry_cases.get("avoid"):
            few_shot_lines.append(f"  注意事项：{'；'.join(industry_cases['avoid'])}")
    
    # 构造策略摘要
    strategy_lines = ["【可选命名策略】"]
    for s in strategies:
        best_example = s["examples"][0]
        strategy_lines.append(f"  {s['id']}. {s['name']}：{s['description']}（如：{best_example['name']}）")
    
    # 构造文化素材摘要（精选）
    culture_lines = ["【文化素材库（可选用）】"]
    for ref in cultural_refs.get("诗经楚辞", [])[:3]:
        culture_lines.append(f"  · {ref['source']}：{ref['text']} → 可用：{'、'.join(ref['usable_words'])}")
    for ref in cultural_refs.get("成语典故", [])[:3]:
        culture_lines.append(f"  · {ref['idiom']} → 可用：{'、'.join(ref['usable_words'])}")
    
    return {
        "strategies": strategies,
        "industry_cases": industry_cases,
        "cultural_refs": cultural_refs,
        "evaluation_dimensions": eval_dims,
        "few_shot_prompt": "\n".join(few_shot_lines),
        "strategy_prompt": "\n".join(strategy_lines),
        "culture_prompt": "\n".join(culture_lines),
    }


def build_evaluation_prompt(eval_dims: dict) -> str:
    """构造评分维度的提示文本"""
    lines = ["请从以下4个维度为每个名称打分（1-10分）："]
    for key, dim in eval_dims.items():
        lines.append(f"  · {dim['name']}：{dim['description']}（{dim['scoring_guide']}）")
    return "\n".join(lines)
