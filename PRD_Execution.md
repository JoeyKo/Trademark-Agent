这是一份为你整理好的 **《基于 LangGraph + Qwen 的商标起名 Agent 开发执行文档》**。
---

# 🛠️ 商标起名 Agent 开发执行文档 (LangGraph + Qwen 版)

## 1. 项目背景

利用 **Qwen (通义千问)** 的强大中文文学修养与 **LangGraph** 的状态机逻辑，构建一个具备“创意生成-合规检查-迭代优化”能力的自动化商标命名工具。

---

## 2. 技术栈架构

* **核心大脑 (LLM):** `Qwen-2.5-72B-Instruct` (推荐使用 ModelScope 免费接口)。
* **逻辑框架:** `LangGraph` (用于处理多轮迭代和条件判断)。
* **开发语言:** Python 3.10+。
* **外部依赖:** * `langchain-openai`: 用于适配 Qwen 的 OpenAI 兼容接口。
* 建议接入 API: 阿里云商标查询 API / 权大师 API (用于真实查重)。



---

## 3. 核心节点 (Nodes) 与工作流

### 节点逻辑定义：

1. **Requirement Analyst (需求分析):** 提取行业类别（45类）、品牌风格、核心关键字。
2. **Creative Generator (创意生成):** Qwen 根据预设的 5 种起名策略（叠词、诗经、谐音等）生成候选方案。
3. **Compliance Checker (合规风控):** * **Level 1:** 关键词过滤（禁用词）。
* **Level 2:** 调用 API 进行同行业重名检索。


4. **Final Optimizer (结果优化):** 若查重未通过且尝试次数 < 3，则将失败原因反馈给生成节点重新起名。

---

## 4. 快速开始 (本地开发配置)

### 4.1 获取免费 API Key

* **推荐方案:** [魔搭社区 ModelScope](https://modelscope.cn/my/myaccesstoken)
* **额度:** 每天免费 2000 次请求。
* **Base URL:** `https://api-inference.modelscope.cn/v1`

### 4.2 环境安装

```bash
pip install langgraph langchain_openai pandas

```

### 4.3 核心代码实现框架 (`agent.py`)

```python
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# 定义状态机数据结构
class AgentState(TypedDict):
    industry: str
    keywords: str
    candidates: List[dict]
    retry_count: int
    error_msg: str

# 初始化 Qwen (ModelScope 免费版)
llm = ChatOpenAI(
    model_name="Qwen/Qwen2.5-72B-Instruct",
    openai_api_base="https://api-inference.modelscope.cn/v1",
    openai_api_key="你的_MODELSCOPE_SDK_TOKEN"
)

# --- 节点函数 ---
def generator_node(state: AgentState):
    prompt = f"作为命名专家，为{state['industry']}行业起3个含'{state['keywords']}'的商标名。"
    if state['error_msg']:
        prompt += f"注意避开以下失败原因：{state['error_msg']}"
    
    res = llm.invoke(prompt)
    # 模拟解析逻辑 (实际开发需用 JSONOutputParser)
    names = res.content.split("\n")[:3] 
    return {"candidates": [{"name": n, "status": "pending"} for n in names], "retry_count": state['retry_count'] + 1}

def checker_node(state: AgentState):
    # 此处应接入真实 API，此处演示模拟逻辑
    for cand in state['candidates']:
        if "阿里" in cand['name']: # 模拟敏感词拦截
            cand['status'] = "fail"
            return {"error_msg": "名称包含敏感词或已注册", "candidates": state['candidates']}
        cand['status'] = "pass"
    return {"candidates": state['candidates'], "error_msg": ""}

# --- 构建图逻辑 ---
workflow = StateGraph(AgentState)
workflow.add_node("generator", generator_node)
workflow.add_node("checker", check_node)

workflow.set_entry_point("generator")
workflow.add_edge("generator", "checker")

def should_continue(state: AgentState):
    if any(c['status'] == "pass" for c in state['candidates']) or state['retry_count'] >= 3:
        return END
    return "generator"

workflow.add_conditional_edges("checker", should_continue)
app = workflow.compile()

```

---

## 5. 待办清单 (Roadmap)

* [ ] **Phase 1:** 完成 LangGraph 基础 Demo，打通 Qwen API。
* [ ] **Phase 2:** 编写详细的 System Prompt，区分“古风、科技、外贸”等风格提示词。
* [ ] **Phase 3:** 调研并接入一个真实的商标查询 API (如阿里云)。
* [ ] **Phase 4:** 增加“含义解析”功能，为每个名字生成 200 字左右的品牌故事。

---

## 6. 注意事项

1. **API 频率:** 虽然魔搭免费，但并发过高会被限流，本地测试建议加 `time.sleep`。
2. **幻觉问题:** Qwen 可能会编造“该商标未被注册”的结论，**必须**以 `checker_node` 返回的真实 API 数据为准。

---