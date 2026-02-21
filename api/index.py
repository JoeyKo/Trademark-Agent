import json
import urllib.parse
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
try:
    # Vercel Serverless 时，项目根目录为执行目录
    from api.agent import AgentState, llm, should_continue, checker_node
except ImportError:
    # 兼容本地 cd api 后执行 uvicorn 的情况
    from agent import AgentState, llm, should_continue, checker_node

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="商标大师起名 API",
    description="基于大模型的商标自动生成与阿里云查询链接构造 API",
    version="1.0.0"
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

def api_generator_node(state: AgentState):
    prompt = f"""你是一位专业的商标命名专家。
    
    请为以下行业和关键词生成3个创意商标名称：
    行业：{state['industry']}
    核心关键词：{state['keywords']}
    
    起名要求：
    1. 必须包含关键词中的核心含义。
    2. 避免使用生僻字和不吉利的词语。
    3. 输出格式：返回一个JSON列表，每个元素包含 name（名称）和 reason（简短理由）。
    """
    
    # 在 API 中我们直接 invoke 获取结果即可，无需 stream 打印到终端
    response = llm.invoke(prompt)
    full_content = response.content

    try:
        content = full_content.strip()
        # 移除Markdown代码块标记（如果有）
        if content.startswith("```"):
            lines = content.split("\n")
            if len(lines) > 2:
                content = "\n".join(lines[1:-1])
        candidates = json.loads(content)
    except Exception as e:
        # 降级处理：简单分割
        candidates = []
        for line in full_content.split("\n"):
            if line.strip():
                candidates.append({"name": line.strip(), "reason": ""})
    
    return {"candidates": candidates, "retry_count": state.get('retry_count', 0) + 1}

# 构建专用于 API 的 graph
api_workflow = StateGraph(AgentState)
api_workflow.add_node("generator", api_generator_node)
api_workflow.add_node("checker", checker_node)
api_workflow.set_entry_point("generator")
api_workflow.add_edge("generator", "checker")
api_workflow.add_conditional_edges("checker", should_continue, ["generator", END])
api_graph = api_workflow.compile()


# ==========================================
# 定义前端输入和输出的规范 (Pydantic Models)
# ==========================================

class GenerateNameRequest(BaseModel):
    industry: str = Field(..., description="行业，例如：新能源汽车", examples=["新能源汽车"])
    keywords: str = Field(..., description="核心关键词，多个词用逗号或空格隔开", examples=["智能、安全、未来"])

class NameCandidate(BaseModel):
    name: str = Field(..., description="生成的商标名称")
    reason: Optional[str] = Field("", description="生成该名称的简短理由")
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

@app.post("/api/v1/generate-names", response_model=GenerateNameResponse, summary="生成商标名称")
async def generate_names_api(request: GenerateNameRequest):
    try:
        initial_input = {
            "industry": request.industry,
            "keywords": request.keywords,
            "candidates": [],
            "retry_count": 0,
            "error_msg": ""
        }
        
        # 执行图
        final_state = api_graph.invoke(initial_input)
        
        # 封装返回给前端的数据结构
        result_candidates = []
        for cand in final_state['candidates']:
            name = cand.get('name', '')
            result_candidates.append(NameCandidate(
                name=name,
                reason=cand.get('reason', ''),
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

if __name__ == "__main__":
    import uvicorn
    # 为了方便测试直接运行此文件
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
