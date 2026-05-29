#!/bin/sh
echo "[GIT]: Updating qwenproxy..."
git pull
echo "[DOCKER]: Stopping and removing qwenproxy..."
docker stop qwenproxy
docker rm qwenproxy
echo "[DOCKER]: Building new qwenproxy..."
docker build -t qwenproxy .
echo "[DOCKER]: Starting qwenproxy..."
docker run -d -p 1234:1234 --name qwenproxy qwenproxy
echo "[QWENPROXY]: Qwenproxy started! Port: 1234"
