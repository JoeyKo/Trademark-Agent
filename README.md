# AI商标起名

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
cd api
uvicorn index:app --port 8000
# 或者使用 python index.py（如有入口代码）
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

---

### ☁️ Vercel 一键部署

本项目支持免服务器（Serverless）云端部署，只需：
1. Fork 或上传此仓库到 GitHub。
2. 在 [Vercel](https://vercel.com/) 控制台中新建项目，导入您的仓库。
3. **关键配置**：在部署前，展开 "Build and Output Settings" 修改参数：
   - **Framework Preset**: 选择 `Astro`
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Output Directory**: `frontend/dist`
4. 展开 **Environment Variables**，添加环境变量：
   - 名称填写：`OPENAI_API_KEY`
   - 填写您的真实 Token 
5. 点击 **Deploy**！🚀

部署完成后，Vercel 会自动将前端打包为静态页面并作为主路由响应，同时将 `/api/v1/*` 的所有请求转发至 Python Serverless 引擎 (`api/index.py`)。双端同享一个域名，解决所有跨域问题！