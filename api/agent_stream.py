import json
import os
import re
from langchain_openai import ChatOpenAI

try:
    from api.rag_retriever import retrieve_context, build_evaluation_prompt
except ImportError:
    from rag_retriever import retrieve_context, build_evaluation_prompt


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


def _build_generation_prompt(industry: str, keywords: str, rag_ctx: dict, feedback: str = "") -> str:
    """构建融合 RAG 知识的多策略生成 Prompt"""
    
    # 基础角色设定
    prompt = f"""你是一位顶级品牌命名大师，精通中国文化、语言学和品牌战略。你曾为多个知名品牌命名。

{rag_ctx['strategy_prompt']}

{rag_ctx['few_shot_prompt']}

{rag_ctx['culture_prompt']}

现在请为以下需求生成 **5个** 高质量商标名称候选方案：
行业：{industry}
核心关键词：{keywords}

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
【输出过程】
请先详细且自然地陈述你的思考和起名灵感过程（不要使用JSON格式，用自然流畅的段落描述你如何运用不同策略）。

【最终结果格式】
思考完成后，务必用 ```json 包裹返回一个JSON列表，每个元素包含：
- name（名称）
- strategy（所用策略名称）
- reason（命名理由，100字以内）
- slogan（品牌标语建议）
"""
    return prompt


def _build_evaluation_prompt(candidates: list, industry: str, keywords: str, eval_dims: dict) -> str:
    """构建自评打分 Prompt"""
    eval_guide = build_evaluation_prompt(eval_dims)
    
    candidates_text = "\n".join([
        f"  {i+1}. {c['name']}（策略：{c.get('strategy', '未知')}）—— {c.get('reason', '')}"
        for i, c in enumerate(candidates)
    ])
    
    return f"""你是一位资深的品牌命名评审专家。请对以下为"{industry}"行业（关键词：{keywords}）生成的商标名称进行严格评分。

候选名称：
{candidates_text}

{eval_guide}

【评分要求】
1. 请严格按照评分标准打分，不要客气或敷衍。低于5分的名称应明确指出问题。
2. 对于每个名称，给出一句简短的评语和改进建议。
3. 综合四维得分计算加权总分：品牌辨识度×0.3 + 文化内涵×0.2 + 行业契合度×0.3 + 注册可行性×0.2

【输出格式】
请用 ```json 包裹返回一个JSON列表，每个元素包含：
- name（名称）
- scores（对象：brand_recognition, cultural_depth, industry_fit, registrability，每个为1-10的整数）
- total_score（加权总分，精确到1位小数）
- comment（简短评语和改进建议）
"""


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


async def stream_generator(industry: str, keywords: str):
    llm = get_llm()
    
    # ==================== 阶段1：RAG 检索 ====================
    yield f"data: {json.dumps({'type': 'status', 'content': '📚 正在检索行业知识库...'})}\n\n"
    
    rag_ctx = retrieve_context(industry, keywords)
    
    has_industry_match = rag_ctx['industry_cases'] is not None
    if has_industry_match:
        yield f"data: {json.dumps({'type': 'status', 'content': f'✅ 已匹配到 {industry} 行业案例库，加载命名策略中...'})}\n\n"
    else:
        yield f"data: {json.dumps({'type': 'status', 'content': '💡 该行业暂无专属案例库，将使用通用命名策略...'})}\n\n"
    
    # ==================== 阶段2：多策略生成（可能迭代） ====================
    max_rounds = 2
    best_candidates = []
    feedback = ""
    
    for round_num in range(1, max_rounds + 1):
        if round_num == 1:
            yield f"data: {json.dumps({'type': 'status', 'content': '🎨 AI 命名大师正在创作中...'})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'status', 'content': f'🔄 第{round_num}轮优化中，针对评估反馈改进...'})}\n\n"
        
        prompt = _build_generation_prompt(industry, keywords, rag_ctx, feedback)
        
        full_content = ""
        try:
            async for chunk in llm.astream(prompt):
                reasoning = chunk.additional_kwargs.get("reasoning_content", "")
                if reasoning:
                    yield f"data: {json.dumps({'type': 'reasoning', 'content': reasoning})}\n\n"
                
                if chunk.content:
                    full_content += chunk.content
                    yield f"data: {json.dumps({'type': 'reasoning', 'content': chunk.content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'生成阶段出错: {str(e)}'})}\n\n"
            return
        
        candidates = _parse_json_from_content(full_content)
        if not candidates:
            yield f"data: {json.dumps({'type': 'error', 'message': '未能解析生成结果，请重试'})}\n\n"
            return
        
        # ==================== 阶段3：自评打分 ====================
        yield f"data: {json.dumps({'type': 'status', 'content': '📊 AI 评审专家正在为候选名称打分...'})}\n\n"
        
        eval_prompt = _build_evaluation_prompt(candidates, industry, keywords, rag_ctx['evaluation_dimensions'])
        
        eval_content = ""
        try:
            async for chunk in llm.astream(eval_prompt):
                # 评分阶段不流式输出到前端，只收集结果
                if chunk.content:
                    eval_content += chunk.content
        except Exception as e:
            print(f"Evaluation error: {e}")
            # 评分失败不阻塞，直接使用未评分的候选
            best_candidates = candidates
            break
        
        scored_candidates = _parse_json_from_content(eval_content)
        
        if scored_candidates:
            # 将评分合并回候选名称
            score_map = {sc['name']: sc for sc in scored_candidates}
            for cand in candidates:
                score_info = score_map.get(cand['name'], {})
                cand['scores'] = score_info.get('scores', {})
                cand['total_score'] = score_info.get('total_score', 0)
                cand['comment'] = score_info.get('comment', '')
            
            # 按总分排序
            candidates.sort(key=lambda x: x.get('total_score', 0), reverse=True)
            
            best_score = candidates[0].get('total_score', 0) if candidates else 0
            
            yield f"data: {json.dumps({'type': 'status', 'content': f'📊 评分完成！最高分：{best_score}/10'})}\n\n"
            
            # 如果最高分 >= 7 或已是最后一轮，取 Top 3
            if best_score >= 7 or round_num >= max_rounds:
                best_candidates = candidates[:3]
                break
            else:
                # 分数不够，构造反馈进行下一轮优化
                low_comments = [
                    f"· {c['name']}（{c.get('total_score', 0)}分）：{c.get('comment', '无评语')}"
                    for c in candidates
                ]
                feedback = "\n".join(low_comments)
                yield f"data: {json.dumps({'type': 'status', 'content': f'⚠️ 当前最高分仅{best_score}分，启动优化重生成...'})}\n\n"
        else:
            # 评分解析失败，使用原始候选
            best_candidates = candidates[:3]
            break
    
    if not best_candidates:
        best_candidates = candidates[:3] if candidates else []
    
    # ==================== 阶段4：合规检查 ====================
    yield f"data: {json.dumps({'type': 'status', 'content': '🔍 正在进行合规检查...'})}\n\n"
    
    try:
        from api.index import build_aliyun_url
    except ImportError:
        from index import build_aliyun_url
    
    failed_names = []
    for cand in best_candidates:
        cand_name = cand.get('name', '')
        if "阿里" in cand_name:
            failed_names.append(cand_name)
            cand['status'] = "fail"
        else:
            cand['status'] = "pass"
        
        cand['query_url'] = build_aliyun_url(cand_name)
    
    yield f"data: {json.dumps({'type': 'result', 'candidates': best_candidates, 'error_msg': f'以下名称包含敏感词或已被注册：{chr(44).join(failed_names)}' if failed_names else ''})}\n\n"
