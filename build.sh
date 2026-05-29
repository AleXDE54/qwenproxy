#!/bin/sh
git pull
docker stop qwenproxy
docker rm qwenproxy
docker build -t qwenproxy .
docker run -d -p 1234:1234 --name qwenproxy qwenproxy
