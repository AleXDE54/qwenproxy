#!/bin/sh
echo "[GIT]: Updating qwenproxy..."
git pull
echo "[DOCKER]: Stopping and removing qwenproxy..."
docker stop qwenproxy
docker rm qwenproxy
echo "[DOCKER]: Building new qwenproxy..."
docker build -t qwenproxy .
echo "[DOCKER]: Starting qwenproxy..."
docker run -d \
  --name qwenproxy \
  --restart unless-stopped \
  -p 1234:1234 \
  -e QWENPROXY_RETRY_MAX_ATTEMPTS=3 \
  -e QWENPROXY_RETRY_MIN_LENGTH=50 \
  -e QWENPROXY_RETRY_DELAY=1 \
  -e QWENPROXY_DEBUG=true \
  qwenproxy
echo "[QWENPROXY]: Qwenproxy started! Port: 1234"
