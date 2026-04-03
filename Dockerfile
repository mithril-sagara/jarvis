FROM python:3.11-slim
RUN apt-get update && apt-get install -y alsa-utils x11-xserver-utils && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# 依存ライブラリのインストール（後で requirements.txt を調整）
RUN pip install --no-cache-dir requests influxdb-client ollama
COPY . .
CMD ["python", "main.py"]
