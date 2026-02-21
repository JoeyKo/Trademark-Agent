### 运行
```bash
python main.py
```
`http://127.0.0.1:8000/api/v1/generate-names` 调用。

### API输入输出规范：

### 请求 (POST)
```json
{
  "industry": "3d行业",
  "keywords": "数字、编辑器、数字世界"
}
```
### 响应
```json
{
  "candidates": [
    {
      "name": "幻维世界",
      "reason": "将'幻维'与'数字世界'结合，富有科技感...",
      "status": "pass",
      "query_url": "https://tm.aliyun.com/channel/search#..."
    },
    ...
  ],
  "retry_count": 1,
  "error_msg": ""
}
```