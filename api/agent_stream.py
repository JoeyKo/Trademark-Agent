import json
import os
from langchain_openai import ChatOpenAI

def get_llm():
    return ChatOpenAI(
        model_name="Qwen/Qwen3-32B",
        openai_api_base="https://api-inference.modelscope.cn/v1",
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        streaming=True,
        extra_body={
            "enable_thinking": True
        },
    )

async def stream_generator(industry: str, keywords: str):
    llm = get_llm()
    prompt = f"""你是一位专业的商标命名专家。

    请为以下行业和关键词生成3个创意商标名称：
    行业：{industry}
    核心关键词：{keywords}

    起名要求：
    1. 必须包含关键词中的核心含义。
    2. 避免使用生僻字和不吉利的词语。
    3. 输出过程：请先详细且自然地陈述你的思考和起名灵感过程（不要使用JSON格式，用自然流畅的段落）。
    4. 最终结果格式：在思考完成之后，请务必用 ```json 包裹返回一个JSON列表，每个元素包含 name（名称）和 reason（简短理由）。
    """

    full_content = ""
    
    try:
        async for chunk in llm.astream(prompt):
            # Check for reasoning chunk
            reasoning = chunk.additional_kwargs.get("reasoning_content", "")
            if reasoning:
                # Need to escape newlines for SSE data payload. Let's just construct a JSON dict
                yield f"data: {json.dumps({'type': 'reasoning', 'content': reasoning})}\n\n"
            
            if chunk.content:
                full_content += chunk.content
                yield f"data: {json.dumps({'type': 'reasoning', 'content': chunk.content})}\n\n"
        
        # After stream finishes, parse content
        import re
        try:
            # Attempt to find JSON array wrapped in markdown code blocks
            match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', full_content, re.DOTALL)
            if match:
                candidates = json.loads(match.group(1))
            else:
                # Fallback: try to find any JSON array bracket
                match = re.search(r'\[\s*{.*?}\s*\]', full_content, re.DOTALL)
                if match:
                    candidates = json.loads(match.group(0))
                else:
                    raise ValueError("No JSON array found in output.")
        except Exception as e:
            print(f"Failed to parse candidates JSON. Error: {e}, Content: {full_content}")
            candidates = []
        
        # Simulate Checker Node
        from api.index import build_aliyun_url
        failed_names = []
        for cand in candidates:
            cand_name = cand.get('name', '')
            if "阿里" in cand_name:
                failed_names.append(cand_name)
                cand['status'] = "fail"
            else:
                cand['status'] = "pass"
            
            cand['query_url'] = build_aliyun_url(cand_name)
                
        yield f"data: {json.dumps({'type': 'result', 'candidates': candidates, 'error_msg': f'以下名称包含敏感词或已被注册：{chr(44).join(failed_names)}' if failed_names else ''})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
