#!/bin/sh
git pull
docker build -t qwenproxy .
docker run -d -p 1234:1234 --name qwenproxy qwenproxy
