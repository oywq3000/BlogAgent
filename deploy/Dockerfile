# 博客 AI Agent 镜像（服务器构建：docker compose up -d --build）
# 服务器需能访问 PyPI；requirements.txt 不变时 pip 层命中 Docker 缓存，构建秒级
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# 镜像内无 .env：全部配置由 compose 的 env_file(.env) 注入
EXPOSE 8001

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
