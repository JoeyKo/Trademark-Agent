# 商标大师起名系统

AI智能驱动的商标起名工具，结合大模型为您一键探索优质品牌方案，并直接生成阿里云商标查重链接。

项目分为两部分：
- **后端**：基于 Python FastAPI + LangChain/LangGraph + Qwen3 大模型构建的智能决策链路。
- **前端**：基于 Astro 开发的现代化玻璃拟态 UI（响应式设计，提供优秀体验）。

---

### 🚀 准备工作 (后端环境)

1. 从 [ModelScope](https://modelscope.cn/) 获取免费的 API Token。
2. 安装依赖：
   ```bash
   pip install modelscope openai langchain-openai langgraph fastapi uvicorn python-dotenv
   ```
3. 在项目根目录下创建 `.env` 文件，并配置你的 Token：
   ```env
   OPENAI_API_KEY=你的真实的Token
   ```

### ⚙️ 运行后端服务 (Agent API)

在终端运行以下命令：
```bash
python main.py
```
> 服务将启动在 `http://127.0.0.1:8000`，同时支持 Swagger 文档访问 (`/docs`)。

---

### 🎨 准备工作 (前端环境)

1. 进入 frontend 目录并安装依赖 (推荐使用 npm 或 pnpm)：
   ```bash
   cd frontend
   npm install
   # 或者使用 pnpm install
   ```

### 💻 运行前端应用

在 `frontend` 目录下运行：
```bash
npm run dev
# 或者使用 pnpm run dev
```
> 前端默认启动在 `http://localhost:4321`，直接在浏览器中打开即可开始体验。