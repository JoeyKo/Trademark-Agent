import json
import urllib.parse
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
try:
    # Vercel Serverless 时，项目根目录为执行目录
    from api.agent import AgentState, llm, should_continue, checker_node, rag_node, evaluator_node, _parse_json_from_content
    from api.agent_stream import stream_generator
    from api.rag_retriever import retrieve_context, build_evaluation_prompt
except ImportError:
    # 兼容本地 cd api 后执行 uvicorn 的情况
    from agent import AgentState, llm, should_continue, checker_node, rag_node, evaluator_node, _parse_json_from_content
    from agent_stream import stream_generator
    from rag_retriever import retrieve_context, build_evaluation_prompt

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="商标起名 API",
    description="基于大模型 + RAG 知识库的商标自动生成与阿里云查询链接构造 API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 重新构建不带流式打印的无声版 Graph (API不需要在终端打印)
# ==========================================

def api_rag_node(state: AgentState):
    """API版 RAG 检索节点（静默）"""
    rag_ctx = retrieve_context(state['industry'], state['keywords'])
    return {"rag_context": rag_ctx}


def api_generator_node(state: AgentState):
    """API版 多策略生成节点（静默）"""
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
上一轮生成的名称评分较低，请针对以下反馈改进：
{feedback}
"""

    prompt += """
【输出格式】
请直接返回用 ```json 包裹的JSON列表，每个元素包含：
- name（名称）
- strategy（所用策略名称）
- reason（命名理由，100字以内）
- slogan（品牌标语建议）
"""
    
    response = llm.invoke(prompt)
    full_content = response.content

    candidates = _parse_json_from_content(full_content)
    if not candidates:
        for line in full_content.split("\n"):
            if line.strip():
                candidates.append({"name": line.strip(), "reason": "", "strategy": "未知", "slogan": ""})
    
    return {"candidates": candidates, "retry_count": state.get('retry_count', 0) + 1}


def api_evaluator_node(state: AgentState):
    """API版 自评打分节点（静默）"""
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
1. 严格按照评分标准打分。
2. 综合四维得分计算加权总分：品牌辨识度×0.3 + 文化内涵×0.2 + 行业契合度×0.3 + 注册可行性×0.2

【输出格式】
请用 ```json 包裹返回一个JSON列表，每个元素包含：
- name（名称）
- scores（对象：brand_recognition, cultural_depth, industry_fit, registrability，每个为1-10的整数）
- total_score（加权总分，精确到1位小数）
- comment（简短评语和改进建议）
"""
    
    response = llm.invoke(eval_prompt)
    scored = _parse_json_from_content(response.content)
    
    if scored:
        score_map = {sc['name']: sc for sc in scored}
        for cand in state['candidates']:
            info = score_map.get(cand['name'], {})
            cand['scores'] = info.get('scores', {})
            cand['total_score'] = info.get('total_score', 0)
            cand['comment'] = info.get('comment', '')
        
        state['candidates'].sort(key=lambda x: x.get('total_score', 0), reverse=True)
        best = state['candidates'][0].get('total_score', 0) if state['candidates'] else 0
        
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


# 构建专用于 API 的 graph
api_workflow = StateGraph(AgentState)
api_workflow.add_node("rag", api_rag_node)
api_workflow.add_node("generator", api_generator_node)
api_workflow.add_node("evaluator", api_evaluator_node)
api_workflow.add_node("checker", checker_node)

api_workflow.set_entry_point("rag")
api_workflow.add_edge("rag", "generator")
api_workflow.add_edge("generator", "evaluator")
api_workflow.add_conditional_edges("evaluator", should_continue, ["generator", "checker"])
api_workflow.add_edge("checker", END)

api_graph = api_workflow.compile()


# ==========================================
# 定义前端输入和输出的规范 (Pydantic Models)
# ==========================================

class GenerateNameRequest(BaseModel):
    industry: str = Field(..., description="行业，例如：新能源汽车", examples=["新能源汽车"])
    keywords: str = Field(..., description="核心关键词，多个词用逗号或空格隔开", examples=["智能、安全、未来"])

class ScoreDetail(BaseModel):
    brand_recognition: int = Field(0, description="品牌辨识度 1-10")
    cultural_depth: int = Field(0, description="文化内涵 1-10")
    industry_fit: int = Field(0, description="行业契合度 1-10")
    registrability: int = Field(0, description="注册可行性 1-10")

class NameCandidate(BaseModel):
    name: str = Field(..., description="生成的商标名称")
    strategy: str = Field("", description="所用命名策略")
    reason: Optional[str] = Field("", description="生成该名称的简短理由")
    slogan: str = Field("", description="品牌标语建议")
    scores: Optional[ScoreDetail] = Field(None, description="四维评分详情")
    total_score: float = Field(0, description="加权总分")
    status: str = Field("pass", description="合规状态，pass或fail")
    query_url: str = Field(..., description="阿里云商标查询的超链接")

class GenerateNameResponse(BaseModel):
    candidates: List[NameCandidate] = Field(..., description="候选商标列表")
    retry_count: int = Field(..., description="生成经过的内部迭代次数")
    error_msg: Optional[str] = Field("", description="错误或警告信息（如有）")


# ==========================================
# API 路由
# ==========================================

def build_aliyun_url(keyword: str) -> str:
    query_dict = {
        "keyword": keyword,
        "searchType": "ALL",
        "pageNum": 1,
        "pageSize": 20,
        "classification": "",
        "product": "",
        "Status": "",
        "ApplyYear": "",
        "applyDateOrder": "",
        "firstAnncDateOrder": "",
        "regDateOrder": "",
        "orderId": "",
        "valid": False,
        "ifPrecise": False,
        "image": ""
    }
    encoded_q = urllib.parse.quote(json.dumps(query_dict, separators=(',', ':')))
    return f"https://tm.aliyun.com/channel/search#/search?q={encoded_q}"

@app.post("/api/v1/generate-names", response_model=GenerateNameResponse, summary="生成商标名称 (旧版弃用)", deprecated=True)
async def generate_names_api(request: GenerateNameRequest):
    try:
        initial_input = {
            "industry": request.industry,
            "keywords": request.keywords,
            "candidates": [],
            "retry_count": 0,
            "error_msg": "",
            "rag_context": None,
            "feedback": "",
            "best_score": 0.0,
        }
        
        # 执行图
        final_state = api_graph.invoke(initial_input)
        
        # 封装返回给前端的数据结构
        result_candidates = []
        for cand in final_state['candidates']:
            name = cand.get('name', '')
            scores_raw = cand.get('scores', {})
            scores = ScoreDetail(**scores_raw) if isinstance(scores_raw, dict) and scores_raw else None
            
            result_candidates.append(NameCandidate(
                name=name,
                strategy=cand.get('strategy', ''),
                reason=cand.get('reason', ''),
                slogan=cand.get('slogan', ''),
                scores=scores,
                total_score=cand.get('total_score', 0),
                status=cand.get('status', 'pass'),
                query_url=build_aliyun_url(name)
            ))
            
        return GenerateNameResponse(
            candidates=result_candidates,
            retry_count=final_state.get('retry_count', 1),
            error_msg=final_state.get('error_msg', "")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内部生成错误: {str(e)}")

@app.post("/api/v1/generate-names-stream", summary="流式生成商标名称（含 RAG + 自评优选）")
async def generate_names_stream_api(request: GenerateNameRequest):
    return StreamingResponse(
        stream_generator(request.industry, request.keywords), 
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    import sys
    import os

    # 自动检测运行目录：如果当前目录下有 index.py，说明在 api/ 内运行
    if os.path.basename(os.getcwd()) == "api":
        app_path = "index:app"
    else:
        # 从项目根目录运行时，将 api/ 加入 sys.path
        sys.path.insert(0, os.path.join(os.getcwd(), "api"))
        app_path = "index:app"

    uvicorn.run(app_path, host="0.0.0.0", port=8000, reload=True)
