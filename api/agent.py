import os
import json
import re
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

try:
    from api.rag_retriever import retrieve_context, build_evaluation_prompt
except ImportError:
    from rag_retriever import retrieve_context, build_evaluation_prompt

# 自动从项目根目录的 .env 文件加载环境变量
load_dotenv()

class AgentState(TypedDict):
    industry: str
    keywords: str
    candidates: List[dict]
    retry_count: int
    error_msg: str
    rag_context: Optional[dict]
    feedback: str
    best_score: float

# 初始化 Qwen (ModelScope 免费版)
llm = ChatOpenAI(
    model_name="Qwen/Qwen3-32B",
    openai_api_base="https://api-inference.modelscope.cn/v1",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    streaming=True,
    extra_body={
      "enable_thinking": True
    },
)


def _parse_json_from_content(content: str) -> list:
    """从 LLM 输出中提取 JSON 列表"""
    try:
        match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r'\[\s*{.*?}\s*\]', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print(f"JSON parse error: {e}")
    return []


# ==================== 节点定义 ====================

def rag_node(state: AgentState):
    """RAG 检索节点：从知识库检索相关上下文"""
    print("\n📚 [RAG 检索中...]")
    rag_ctx = retrieve_context(state['industry'], state['keywords'])
    has_match = rag_ctx['industry_cases'] is not None
    print(f"{'✅ 匹配到行业案例库' if has_match else '💡 使用通用命名策略'}")
    return {"rag_context": rag_ctx}


def generator_node(state: AgentState):
    """多策略生成节点：融合 RAG 上下文生成5个候选名称"""
    rag_ctx = state.get('rag_context', {})
    feedback = state.get('feedback', '')
    
    prompt = f"""你是一位顶级品牌命名大师，精通中国文化、语言学和品牌战略。

{rag_ctx.get('strategy_prompt', '')}

{rag_ctx.get('few_shot_prompt', '')}

{rag_ctx.get('culture_prompt', '')}

现在请为以下需求生成 **5个** 高质量商标名称候选方案：
行业：{state['industry']}
核心关键词：{state['keywords']}

【起名要求】
1. 每个名称必须包含关键词的核心含义或精神内核，但不要直接堆砌关键词。
2. 必须至少使用3种不同的命名策略（谐音双关/古典意象/中外融合/叠词叠韵/行业造词），避免5个名称风格雷同。
3. 名称长度控制在2-4个汉字，避免生僻字、不吉利谐音和笔画过多的字。
4. 每个名称要附带：所用策略、命名理由（100字以内）、以及一句品牌标语建议。
5. 追求"过目不忘"的效果：名称应朗朗上口、有画面感、有情感共鸣。
"""

    if feedback:
        prompt += f"""
【改进要求】
上一轮生成的名称评分较低，以下是评估反馈，请针对性改进：
{feedback}
请务必避免上一轮的问题，大胆尝试新的方向。
"""

    prompt += """
【输出格式】
请直接返回用 ```json 包裹的JSON列表，每个元素包含：
- name（名称）
- strategy（所用策略名称）
- reason（命名理由，100字以内）
- slogan（品牌标语建议）
"""
    
    print(f"\n🎨 [第{state.get('retry_count', 0) + 1}轮生成中...]")
    full_content = ""
    
    for chunk in llm.stream(prompt):
        reasoning = chunk.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            print(reasoning, end="", flush=True)
        if chunk.content:
            full_content += chunk.content
            print(chunk.content, end="", flush=True)
            
    print("\n✨ [生成结束]\n")
    
    candidates = _parse_json_from_content(full_content)
    if not candidates:
        # 降级处理
        for line in full_content.split("\n"):
            if line.strip():
                candidates.append({"name": line.strip(), "reason": "", "strategy": "未知", "slogan": ""})
    
    return {"candidates": candidates, "retry_count": state.get('retry_count', 0) + 1}


def evaluator_node(state: AgentState):
    """自评打分节点：对候选名称从4个维度打分"""
    rag_ctx = state.get('rag_context', {})
    eval_dims = rag_ctx.get('evaluation_dimensions', {})
    
    if not eval_dims or not state['candidates']:
        return {"best_score": 0}
    
    eval_guide = build_evaluation_prompt(eval_dims)
    candidates_text = "\n".join([
        f"  {i+1}. {c['name']}（策略：{c.get('strategy', '未知')}）—— {c.get('reason', '')}"
        for i, c in enumerate(state['candidates'])
    ])
    
    eval_prompt = f"""你是一位资深的品牌命名评审专家。请对以下为"{state['industry']}"行业（关键词：{state['keywords']}）生成的商标名称进行严格评分。

候选名称：
{candidates_text}

{eval_guide}

【评分要求】
1. 严格按照评分标准打分，不要敷衍。
2. 对于每个名称，给出简短评语和改进建议。
3. 综合四维得分计算加权总分：品牌辨识度×0.3 + 文化内涵×0.2 + 行业契合度×0.3 + 注册可行性×0.2

【输出格式】
请用 ```json 包裹返回一个JSON列表，每个元素包含：
- name（名称）
- scores（对象：brand_recognition, cultural_depth, industry_fit, registrability，每个为1-10的整数）
- total_score（加权总分，精确到1位小数）
- comment（简短评语和改进建议）
"""
    
    print("\n📊 [评分中...]")
    eval_content = ""
    for chunk in llm.stream(eval_prompt):
        if chunk.content:
            eval_content += chunk.content
    
    scored = _parse_json_from_content(eval_content)
    
    if scored:
        score_map = {sc['name']: sc for sc in scored}
        for cand in state['candidates']:
            info = score_map.get(cand['name'], {})
            cand['scores'] = info.get('scores', {})
            cand['total_score'] = info.get('total_score', 0)
            cand['comment'] = info.get('comment', '')
        
        state['candidates'].sort(key=lambda x: x.get('total_score', 0), reverse=True)
        best = state['candidates'][0].get('total_score', 0)
        print(f"📊 最高分：{best}/10")
        
        # 如果分数不够，构造反馈
        if best < 7:
            feedback_lines = [
                f"· {c['name']}（{c.get('total_score', 0)}分）：{c.get('comment', '')}"
                for c in state['candidates']
            ]
            return {
                "candidates": state['candidates'][:3],
                "best_score": best,
                "feedback": "\n".join(feedback_lines),
            }
        
        return {"candidates": state['candidates'][:3], "best_score": best, "feedback": ""}
    
    return {"best_score": 0, "feedback": ""}


def checker_node(state: AgentState):
    """合规检查节点"""
    failed_names = []
    for cand in state['candidates']:
        if "阿里" in cand.get('name', ''):
            failed_names.append(cand['name'])
            cand['status'] = "fail"
        else:
            cand['status'] = "pass"
    
    if failed_names:
        return {
            "error_msg": f"以下名称包含敏感词或已被注册：{', '.join(failed_names)}",
            "candidates": state['candidates']
        }
    else:
        return {"error_msg": "", "candidates": state['candidates']}


# --- 构建图逻辑 ---
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("rag", rag_node)
workflow.add_node("generator", generator_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("checker", checker_node)

# 设置入口点和边
workflow.set_entry_point("rag")
workflow.add_edge("rag", "generator")
workflow.add_edge("generator", "evaluator")

# 条件边：评分是否足够或已达到重试上限
def should_continue(state: AgentState):
    """判断是否需要优化重生成"""
    best_score = state.get('best_score', 0)
    retry_count = state.get('retry_count', 0)
    
    if best_score >= 7 or retry_count >= 2:
        return "checker"
    else:
        return "generator"

workflow.add_conditional_edges(
    "evaluator",
    should_continue,
    ["generator", "checker"]
)

workflow.add_edge("checker", END)

# 编译图
graph = workflow.compile()
