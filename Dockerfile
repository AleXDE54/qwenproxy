FROM python:3.11-slim
WORKDIR /qwenproxy
COPY qwenproxy_fast.py /qwenproxy/main.py
COPY req.txt /qwenproxy/req.txt
RUN pip install --no-cache-dir -r req.txt
CMD ["python", "main.py"]
