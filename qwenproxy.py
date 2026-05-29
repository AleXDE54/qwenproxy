#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import re
import uuid
import os
import sys
import time
from typing import Literal, Optional, Dict, Any, AsyncGenerator, List, Union
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from g4f.Provider.qwen.cookie_generator import generate_cookies
    has_g4f = True
except ImportError:
    has_g4f = False

# ==================== Configuration ====================

class Config:
    HOST = os.getenv("QWENPROXY_HOST", "0.0.0.0")
    PORT = int(os.getenv("QWENPROXY_PORT", 1234))
    DEBUG = os.getenv("QWENPROXY_DEBUG", "false").lower() == "true"
    API_KEY = os.getenv("QWENPROXY_API_KEY", None)
    PROXY = os.getenv("QWENPROXY_PROXY", None)
    TIMEOUT = int(os.getenv("QWENPROXY_TIMEOUT", 300))
    BASE_URL = "https://chat.qwen.ai"
    DEFAULT_MODEL = "qwen3-235b-a22b"
    MIDTOKEN_REFRESH_INTERVAL = 3600
    SERVER_SYSTEM_PROMPT = os.getenv("QWENPROXY_SYSTEM_PROMPT", "")
    
    RETRY_MAX_ATTEMPTS = int(os.getenv("QWENPROXY_RETRY_MAX_ATTEMPTS", "3"))
    RETRY_MIN_LENGTH = int(os.getenv("QWENPROXY_RETRY_MIN_LENGTH", "50"))
    RETRY_DELAY = float(os.getenv("QWENPROXY_RETRY_DELAY", "1"))

config = Config()

# ==================== Models ====================

all_models = [
    'qwen3.6-plus', 'qwen3.6-max-preview', 'qwen3.5-plus', 'qwen3.5-omni-plus',
    'qwen3.6-35b-a3b', 'qwen3.5-flash', 'qwen3.5-max-2026-03-08',
    'qwen3.6-plus-preview', 'qwen3.5-397b-a17b', 'qwen3.5-122b-a10b',
    'qwen3.5-omni-flash', 'qwen3.5-27b', 'qwen3.5-35b-a3b',
    'qwen-max-latest'
]

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

    def get_content_str(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return " ".join(
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in self.content
        )

class ChatCompletionRequest(BaseModel):
    model: str = config.DEFAULT_MODEL
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 0.8
    max_tokens: Optional[int] = 4096
    stream: bool = False
    stop: Optional[list[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    chat_type: Literal["t2t", "search", "artifacts", "web_dev", "deep_research", "t2i", "image_edit", "t2v"] = "t2t"
    aspect_ratio: Optional[Literal["1:1", "4:3", "3:4", "16:9", "9:16"]] = None

# ==================== Helper Functions ====================

def format_conversation(messages: List[ChatMessage], server_system_prompt: str = "") -> str:
    """Convert the full message list into a single prompt string."""
    lines = []
    if server_system_prompt:
        lines.append(f"SYSTEM: {server_system_prompt}")
    for msg in messages:
        role = "SYSTEM" if msg.role == "system" else "AI" if msg.role == "assistant" else "USER"
        lines.append(f"{role}: {msg.get_content_str()}")
    return "\n".join(lines)

# ==================== Qwen Client ====================

class QwenClient:
    _midtoken: str = None
    _midtoken_last_refresh: int = 0
    _midtoken_lock = asyncio.Lock()

    @classmethod
    def _get_headers(cls, token: str = None) -> dict:
        data = generate_cookies() if has_g4f else {"ssxmod_itna": "", "ssxmod_itna2": ""}
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Origin': config.BASE_URL,
            'Referer': f'{config.BASE_URL}/',
            'Content-Type': 'application/json',
            'Cookie': f'ssxmod_itna={data["ssxmod_itna"]};ssxmod_itna2={data["ssxmod_itna2"]}',
            'X-Source': 'web'
        }
        if token:
            headers['Authorization'] = f'Bearer {token}'
        return headers

    @classmethod
    async def _get_req_headers(cls, session: aiohttp.ClientSession) -> dict:
        async with cls._midtoken_lock:
            current_time = time.time()
            if not cls._midtoken or (current_time - cls._midtoken_last_refresh) > config.MIDTOKEN_REFRESH_INTERVAL:
                try:
                    async with session.get('https://sg-wum.alibaba.com/w/wu.json', proxy=config.PROXY) as r:
                        text = await r.text()
                        match = re.search(r"(?:umx\.wu|__fycb)\('([^']+)'\)", text)
                        if match:
                            cls._midtoken = match.group(1)
                            cls._midtoken_last_refresh = int(current_time)
                        else:
                            raise RuntimeError("Failed to extract bx-umidtoken.")
                except Exception as e:
                    raise RuntimeError(f"Failed to get midtoken: {e}")
        req_headers = dict(session.headers)
        req_headers['bx-umidtoken'] = cls._midtoken
        req_headers['bx-v'] = '2.5.31'
        return req_headers

    @classmethod
    async def create_chat(
        cls,
        model: str,
        messages: List[ChatMessage],
        temperature: float,
        top_p: float,
        max_tokens: int,
        presence_penalty: float,
        frequency_penalty: float,
        reasoning_effort: str = "medium",
        chat_type: str = "t2t",
        aspect_ratio: str = None,
        stop: Optional[List[str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream raw chunks from Qwen."""
        enable_thinking = reasoning_effort in ("medium", "high")
        thinking_mode = "Auto" if enable_thinking else "Fast"
        prompt = format_conversation(messages, config.SERVER_SYSTEM_PROMPT)

        async with aiohttp.ClientSession(headers=cls._get_headers()) as session:
            req_headers = await cls._get_req_headers(session)

            # Create chat session
            chat_payload = {
                "title": "New Chat",
                "models": [model],
                "chat_mode": "normal",
                "chat_type": chat_type,
                "timestamp": int(time.time() * 1000)
            }
            async with session.post(
                f'{config.BASE_URL}/api/v2/chats/new',
                json=chat_payload,
                headers=req_headers,
                proxy=config.PROXY
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Chat creation failed: {resp.status}")
                data = await resp.json()
                if not data.get('success') or not data.get('data', {}).get('id'):
                    raise RuntimeError(f"Chat creation error: {data}")
                chat_id = data['data']['id']

            # Feature config
            feature_config = {
                "auto_thinking": enable_thinking,
                "thinking_mode": thinking_mode,
                "thinking_enabled": enable_thinking,
                "output_schema": "phase",
                "research_mode": "normal",
                "auto_search": True
            } if enable_thinking else {
                "thinking_enabled": enable_thinking,
                "output_schema": "phase",
                "thinking_budget": 81920
            }

            # Generation config (top-level)
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
            }
            if stop:
                generation_config["stop"] = stop

            # Message payload
            msg_payload = {
                "stream": True,
                "incremental_output": True,
                "chat_id": chat_id,
                "chat_mode": "normal",
                "model": model,
                "parent_id": None,
                "messages": [{
                    "fid": str(uuid.uuid4()),
                    "parentId": None,
                    "childrenIds": [],
                    "role": "user",
                    "content": prompt,
                    "user_action": "chat",
                    "files": [],
                    "models": [model],
                    "chat_type": chat_type,
                    "feature_config": feature_config,
                    "sub_chat_type": chat_type,
                }],
                "generation_config": generation_config   # ✅ TOP LEVEL
            }
            if aspect_ratio:
                msg_payload["size"] = aspect_ratio

            async with session.post(
                f'{config.BASE_URL}/api/v2/chat/completions?chat_id={chat_id}',
                json=msg_payload,
                headers=req_headers,
                proxy=config.PROXY,
                timeout=aiohttp.ClientTimeout(total=config.TIMEOUT)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Completion failed: {resp.status} - {text}")

                buffer = ""
                async for chunk in resp.content.iter_any():
                    buffer += chunk.decode('utf-8')
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith('data:'):
                            data_str = line[5:].strip()
                        else:
                            data_str = line
                        if data_str == '[DONE]':
                            continue
                        try:
                            parsed = json.loads(data_str)
                            if config.DEBUG:
                                print(f"[RAW] {json.dumps(parsed, ensure_ascii=False)[:500]}")
                            yield parsed
                        except json.JSONDecodeError:
                            if config.DEBUG:
                                print(f"[SKIP] {data_str[:100]}")
                            continue
                if buffer.strip():
                    try:
                        parsed = json.loads(buffer)
                        if config.DEBUG:
                            print(f"[RAW final] {json.dumps(parsed, ensure_ascii=False)[:500]}")
                        yield parsed
                    except:
                        pass

    @classmethod
    async def complete_chat_with_retry(
        cls,
        model: str,
        messages: List[ChatMessage],
        temperature: float,
        top_p: float,
        max_tokens: int,
        presence_penalty: float,
        frequency_penalty: float,
        reasoning_effort: str = "medium",
        chat_type: str = "t2t",
        aspect_ratio: str = None,
        stop: Optional[List[str]] = None,
    ) -> tuple[str, str, Optional[dict], str]:
        """Non‑streaming with retry if response too short."""
        attempt = 1
        last_error = None
        t = temperature
        r = reasoning_effort

        while attempt <= config.RETRY_MAX_ATTEMPTS:
            full_content = ""
            full_reasoning = ""
            usage = None
            finish_reason = "stop"

            try:
                async for raw in cls.create_chat(
                    model=model, messages=messages, temperature=t, top_p=top_p,
                    max_tokens=max_tokens, presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty, reasoning_effort=r,
                    chat_type=chat_type, aspect_ratio=aspect_ratio, stop=stop
                ):
                    # Extract content
                    content = ""
                    reasoning = ""
                    if "content" in raw:
                        content = raw["content"]
                    if "reasoning_content" in raw:
                        reasoning = raw["reasoning_content"]
                    if not content and "choices" in raw and raw["choices"]:
                        delta = raw["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        reasoning = delta.get("reasoning_content", "")
                    if not content and "text" in raw:
                        content = raw["text"]
                    full_content += content
                    full_reasoning += reasoning

                    if "choices" in raw and raw["choices"]:
                        fr = raw["choices"][0].get("finish_reason")
                        if fr:
                            finish_reason = fr
                    if "usage" in raw:
                        usage = raw["usage"]

                total_len = len(full_content) + len(full_reasoning)
                if total_len >= config.RETRY_MIN_LENGTH:
                    return full_content, full_reasoning, usage, finish_reason
                else:
                    if config.DEBUG:
                        print(f"⚠️ Short response ({total_len} chars), retry {attempt}/{config.RETRY_MAX_ATTEMPTS}")
                    t = max(0.1, t * 0.7)
                    r = "low"
                    await asyncio.sleep(config.RETRY_DELAY)
                    attempt += 1
            except Exception as e:
                last_error = e
                if config.DEBUG:
                    print(f"❌ Attempt {attempt} failed: {e}")
                await asyncio.sleep(config.RETRY_DELAY)
                attempt += 1

        if last_error:
            raise last_error
        return full_content, full_reasoning, usage, finish_reason

# ==================== FastAPI App ====================

app = FastAPI(title="QwenProxy Fixed", version="3.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

async def verify_api_key(x_api_key: Optional[str] = None):
    if config.API_KEY and (not x_api_key or x_api_key != config.API_KEY):
        raise HTTPException(401, "Invalid API key")
    return True

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in all_models]}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, api_key: Optional[str] = Depends(verify_api_key)):
    if request.model not in all_models:
        raise HTTPException(400, f"Model '{request.model}' not found")

    conv_id = str(uuid.uuid4())
    max_tokens = request.max_tokens or 4096

    if request.stream:
        async def stream_generator():
            created = int(time.time())
            finished = False
            try:
                async for raw in QwenClient.create_chat(
                    model=request.model,
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=max_tokens,
                    presence_penalty=request.presence_penalty,
                    frequency_penalty=request.frequency_penalty,
                    reasoning_effort=request.reasoning_effort,
                    chat_type=request.chat_type,
                    aspect_ratio=request.aspect_ratio,
                    stop=request.stop,
                ):
                    content = ""
                    reasoning = ""
                    finish_reason = None

                    # Extract fields
                    if "content" in raw:
                        content = raw["content"]
                    if "reasoning_content" in raw:
                        reasoning = raw["reasoning_content"]
                    if not content and "choices" in raw and raw["choices"]:
                        delta = raw["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        reasoning = delta.get("reasoning_content", "")
                        finish_reason = raw["choices"][0].get("finish_reason")
                    if not content and "text" in raw:
                        content = raw["text"]
                    # Also check for top‑level finish_reason
                    if not finish_reason and "finish_reason" in raw:
                        finish_reason = raw["finish_reason"]

                    delta = {}
                    if reasoning:
                        delta["reasoning_content"] = reasoning
                    if content:
                        delta["content"] = content

                    # Send chunk if there is content or a finish reason
                    if delta or finish_reason:
                        chunk = {
                            "id": conv_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    if finish_reason:
                        finished = True
                        break

                # Always send a final stop chunk if not already done
                if not finished:
                    final_chunk = {
                        "id": conv_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                if config.DEBUG:
                    import traceback
                    traceback.print_exc()
                error_chunk = {
                    "error": {"message": str(e), "type": "server_error"}
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        try:
            content, reasoning, usage, finish_reason = await QwenClient.complete_chat_with_retry(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=max_tokens,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                reasoning_effort=request.reasoning_effort,
                chat_type=request.chat_type,
                aspect_ratio=request.aspect_ratio,
                stop=request.stop,
            )
            msg = {"role": "assistant", "content": content}
            if reasoning:
                msg["reasoning_content"] = reasoning
            return {
                "id": conv_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
                "usage": usage or {}
            }
        except Exception as e:
            raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

def run_server():
    print(f"🚀 QwenProxy Fixed v3.4.0 on {config.HOST}:{config.PORT}")
    print(f"   Debug: {config.DEBUG}")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")

if __name__ == "__main__":
    run_server()
