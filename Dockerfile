FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_OFFLINE=1

# 安装 Python 3.10 和 pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ============================================================
# 第 1 层：只安装 torch（最大，单独一层以便缓存）
# ============================================================
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 \
    torch==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu118

# ============================================================
# 第 2 层：安装其他大型依赖（sentence-transformers 等）
# ============================================================
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 \
    sentence-transformers \
    transformers \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ============================================================
# 第 3 层：安装剩余所有依赖（从 requirements.txt）
# ============================================================
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 \
    -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ============================================================
# 复制应用代码和数据目录
# ============================================================
COPY app.py ./
COPY chroma_db ./chroma_db
COPY hf_cache ./hf_cache
COPY lora_adapter ./lora_adapter
COPY knowledge.txt ./

EXPOSE 7860
CMD ["python", "app.py"]