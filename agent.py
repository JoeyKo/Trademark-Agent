import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 自动从项目根目录的 .env 文件加载环境变量
load_dotenv()

class AgentState(TypedDict):
    industry: str
    keywords: str
    candidates: List[dict]
    retry_count: int
    error_msg: str

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

def generator_node(state: AgentState):
    prompt = f"""你是一位专业的商标命名专家。
    
    请为以下行业和关键词生成3个创意商标名称：
    行业：{state['industry']}
    核心关键词：{state['keywords']}
    
    起名要求：
    1. 必须包含关键词中的核心含义。
    2. 避免使用生僻字和不吉利的词语。
    3. 输出格式：返回一个JSON列表，每个元素包含 name（名称）和 reason（简短理由）。
    """
    
    print("\n🤖 [模型正在思考...]")
    full_content = ""
    
    # 使用流式输出来打印思考过程
    for chunk in llm.stream(prompt):
        # 常见的带有思维链的模型（如 DeepSeek-R1 / QwQ 等）会将思考过程存在 reasoning_content 中
        reasoning = chunk.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            print(reasoning, end="", flush=True)
            
        # 获取最终回复的文本（如果思考过程是带 <think> 标签混杂在 content 中也会被拼接到这里）
        if chunk.content:
            full_content += chunk.content
            # 打印模型输出的全部内容（因为有些模型的思考过程直接混在 content 中）
            print(chunk.content, end="", flush=True)
            
    print("\n✨ [思考结束]\n")
    
    # 简单的解析逻辑（实际应用中建议使用 PydanticOutputParser）
    try:
        # 尝试直接解析JSON
        import json
        content = full_content.strip()
        
        # 移除Markdown代码块标记（如果有）
        if content.startswith("```"):
            lines = content.split("\n")
            if len(lines) > 2:
                content = "\n".join(lines[1:-1])
        
        candidates = json.loads(content)
    except:
        # 降级处理：简单分割
        candidates = []
        for line in full_content.split("\n"):
            if line.strip():
                candidates.append({"name": line.strip(), "reason": ""})
    
    return {"candidates": candidates, "retry_count": state.get('retry_count', 0) + 1}

def checker_node(state: AgentState):
    """合规检查节点（模拟）"""
    # 实际应用中，这里会调用商标查询API
    # 为了演示，我们模拟检查关键词是否包含“阿里”
    
    failed_names = []
    for cand in state['candidates']:
        if "阿里" in cand['name']:
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
workflow.add_node("generator", generator_node)
workflow.add_node("checker", checker_node)

# 设置入口点
workflow.set_entry_point("generator")
# 设置边
workflow.add_edge("generator", "checker")

# 条件边：检查是否有通过的名称
def should_continue(state: AgentState):
    """判断是否继续循环"""
    # 如果有通过的名称，或者尝试次数已达上限，则停止
    if any(c['status'] == "pass" for c in state['candidates']) or state.get('retry_count', 0) >= 3:
        return END
    else:
        return "generator"

workflow.add_conditional_edges(
    "checker",
    should_continue,
    ["generator", END]
)

# 编译图
graph = workflow.compile()


