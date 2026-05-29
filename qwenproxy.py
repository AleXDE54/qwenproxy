#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import uuid
import os
import sys
import time as time_module
from time import time
from typing import Literal, Optional, Dict, Any, AsyncGenerator, List, Union
from urllib.parse import quote
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

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
    SINGLE_PROMPT_MODE = os.getenv("QWENPROXY_SINGLE_PROMPT", "false").lower() == "true"
    SERVER_SYSTEM_PROMPT = os.getenv("QWENPROXY_SYSTEM_PROMPT", "")
    
    # ----- FALLBACK / RETRY SETTINGS -----
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
    'qwen3-max-2026-01-23', 'qwen-plus-2025-07-28', 'qwen3-coder-plus',
    'qwen3-vl-plus', 'qwen3-omni-flash-2025-12-01', 'qwen-max-latest'
]

# ==================== Pydantic Models ====================

def normalize_content(content: Union[str, List[Dict[str, Any]]]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in content
        )
    return str(content)

class ChatMessage(BaseModel):
    role: str
    content: str | list[dict]

    def get_content_str(self) -> str:
        return normalize_content(self.content)

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
    use_single_prompt: Optional[bool] = None

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "qwen"

class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]

# ==================== Helper Functions ====================

def get_last_user_message(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.get_content_str()
    return ""

def merge_system_prompts(messages: List[ChatMessage], server_system_prompt: str) -> List[ChatMessage]:
    if not server_system_prompt:
        return messages
    new = [ChatMessage(role="system", content=server_system_prompt)]
    for msg in messages:
        if msg.role == "system":
            new[0] = ChatMessage(role="system", content=f"{server_system_prompt}\n\n{msg.get_content_str()}")
        else:
            new.append(msg)
    return new

def messages_to_single_prompt(messages: List[ChatMessage]) -> str:
    return "\n".join(
        f"{('SYSTEM' if m.role == 'system' else 'AI' if m.role == 'assistant' else 'USER')}: {m.get_content_str()}"
        for m in messages
    )

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
            'Accept-Language': 'en-US,en;q=0.5',
            'Origin': config.BASE_URL,
            'Referer': f'{config.BASE_URL}/',
            'Content-Type': 'application/json',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Connection': 'keep-alive',
            'X-Requested-With': 'XMLHttpRequest',
            'Cookie': f'ssxmod_itna={data["ssxmod_itna"]};ssxmod_itna2={data["ssxmod_itna2"]}',
            'X-Source': 'web'
        }
        if token:
            headers['Authorization'] = f'Bearer {token}'
        return headers

    @classmethod
    async def _get_req_headers(cls, session: aiohttp.ClientSession) -> dict:
        async with cls._midtoken_lock:
            current_time = time()
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
        messages: list[ChatMessage],
        temperature: float,
        top_p: float,
        max_tokens: int,
        presence_penalty: float,
        frequency_penalty: float,
        reasoning_effort: str = "medium",
        chat_type: str = "t2t",
        aspect_ratio: str = None,
        use_single_prompt: bool = False,
        server_system_prompt: str = ""
    ) -> AsyncGenerator[dict, None]:
        """Streams raw Qwen API chunks (always streaming internally)."""
        enable_thinking = reasoning_effort in ("medium", "high")
        thinking_mode: Literal["Auto", "Thinking", "Fast"] = "Auto" if enable_thinking else "Fast"
        auto_thinking = thinking_mode == "Auto"

        if use_single_prompt:
            merged_messages = merge_system_prompts(messages, server_system_prompt)
            prompt = messages_to_single_prompt(merged_messages)
        else:
            prompt = get_last_user_message(messages)

        message_id = str(uuid.uuid4())

        async with aiohttp.ClientSession(headers=cls._get_headers()) as session:
            req_headers = await cls._get_req_headers(session)

            # Create a new chat
            chat_payload = {
                "title": "New Chat",
                "models": [model],
                "chat_mode": "normal",
                "chat_type": chat_type,
                "timestamp": int(time() * 1000)
            }
            async with session.post(
                f'{config.BASE_URL}/api/v2/chats/new',
                json=chat_payload,
                headers=req_headers,
                proxy=config.PROXY
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Failed to create chat: {resp.status}")
                data = await resp.json()
                if not (data.get('success') and data.get('data', {}).get('id')):
                    raise RuntimeError(f"Failed to create chat: {data}")
                chat_id = data['data']['id']

            # Prepare feature config
            feature_config = {
                "auto_thinking": auto_thinking,
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

            # Send message with all sampling parameters
            msg_payload = {
                "stream": True,
                "incremental_output": True,
                "chat_id": chat_id,
                "chat_mode": "normal",
                "model": model,
                "parent_id": None,
                "messages": [{
                    "fid": message_id,
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
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty
                }]
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
                    raise RuntimeError(f"Completion failed: {resp.status}")

                buffer = ""
                async for chunk in resp.content.iter_any():
                    buffer += chunk.decode('utf-8')
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line.startswith('data:'):
                            data_str = line[5:].strip()
                            if data_str == '[DONE]':
                                continue
                            try:
                                yield json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                if buffer.strip():
                    if buffer.startswith('data:'):
                        data_str = buffer[5:].strip()
                        if data_str != '[DONE]':
                            try:
                                yield json.loads(data_str)
                            except:
                                pass

    # ----- helper to collect full response (non-stream) with retries -----
    @classmethod
    async def complete_chat_with_retry(
        cls,
        model: str,
        messages: list[ChatMessage],
        temperature: float,
        top_p: float,
        max_tokens: int,
        presence_penalty: float,
        frequency_penalty: float,
        reasoning_effort: str = "medium",
        chat_type: str = "t2t",
        aspect_ratio: str = None,
        use_single_prompt: bool = False,
        server_system_prompt: str = ""
    ) -> tuple[str, str, Optional[dict], str]:
        """
        Returns (full_content, full_reasoning, usage, finish_reason)
        with automatic retry if response is too short.
        """
        attempt = 1
        last_error = None

        while attempt <= config.RETRY_MAX_ATTEMPTS:
            full_content = ""
            full_reasoning = ""
            usage = None
            finish_reason = "stop"

            try:
                async for raw_chunk in cls.create_chat(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    reasoning_effort=reasoning_effort,
                    chat_type=chat_type,
                    aspect_ratio=aspect_ratio,
                    use_single_prompt=use_single_prompt,
                    server_system_prompt=server_system_prompt
                ):
                    phase = raw_chunk.get("phase")
                    if not phase:
                        choices = raw_chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            phase = delta.get("phase")

                    content = raw_chunk.get("content", "")
                    if not content:
                        choices = raw_chunk.get("choices", [])
                        if choices:
                            content = choices[0].get("delta", {}).get("content", "")

                    if phase == "think":
                        full_reasoning += content
                    else:
                        full_content += content

                    if "choices" in raw_chunk and raw_chunk["choices"]:
                        if raw_chunk["choices"][0].get("finish_reason"):
                            finish_reason = raw_chunk["choices"][0]["finish_reason"]

                    if "usage" in raw_chunk:
                        usage = raw_chunk["usage"]

                # Check if response is acceptable
                total_len = len(full_content) + len(full_reasoning)
                if total_len >= config.RETRY_MIN_LENGTH:
                    # Good response
                    return full_content, full_reasoning, usage, finish_reason
                else:
                    # Too short – retry
                    if config.DEBUG:
                        print(f"⚠️ Response too short ({total_len} chars), attempt {attempt}/{config.RETRY_MAX_ATTEMPTS}. Retrying in {config.RETRY_DELAY}s...")
                    await asyncio.sleep(config.RETRY_DELAY)
                    attempt += 1
                    continue

            except Exception as e:
                last_error = e
                if config.DEBUG:
                    print(f"❌ Attempt {attempt} failed: {e}. Retrying in {config.RETRY_DELAY}s...")
                await asyncio.sleep(config.RETRY_DELAY)
                attempt += 1
                continue

        # All attempts exhausted
        if last_error:
            raise last_error
        else:
            # Return what we have (even if short) as a last resort
            return full_content, full_reasoning, usage, finish_reason

# ==================== FastAPI Application ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="QwenProxy Fast", description="High-performance Qwen AI proxy with fallback", version="3.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def verify_api_key(x_api_key: Optional[str] = None):
    if config.API_KEY:
        if not x_api_key or x_api_key != config.API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return True

@app.get("/")
async def root():
    return {"service": "QwenProxy Fast", "version": "3.1.0", "status": "running", "retry_enabled": True}

@app.get("/v1")
async def v1_root():
    return HTMLResponse('''
    <html>
        <head><title>QwenProxy Fast API</title></head>
        <body>
            <h1>QwenProxy Fast API</h1>
            <p>OpenAI-compatible API for Qwen models with automatic retry on short responses</p>
            <ul>
                <li><a href="/v1/models">/v1/models</a> - List available models</li>
                <li><a href="/v1/chat/completions">/v1/chat/completions</a> - Chat completions</li>
            </ul>
        </body>
    </html>
    ''')

@app.get("/v1/models")
async def list_models():
    return ModelList(data=[ModelCard(id=m) for m in all_models])

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, api_key: Optional[str] = Depends(verify_api_key)):
    model = request.model
    messages = request.messages

    if model not in all_models:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not found.")

    conversation_id = str(uuid.uuid4())
    use_single_prompt = request.use_single_prompt if request.use_single_prompt is not None else config.SINGLE_PROMPT_MODE
    max_tokens = request.max_tokens if request.max_tokens is not None else 4096

    # For streaming, we cannot easily retry because data already sent.
    # We'll log a warning if response seems short (but no retry).
    if request.stream:
        async def generate_stream():
            created = int(time())
            full_content_buffer = ""
            try:
                async for raw_chunk in QwenClient.create_chat(
                    model=model,
                    messages=messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=max_tokens,
                    presence_penalty=request.presence_penalty,
                    frequency_penalty=request.frequency_penalty,
                    reasoning_effort=request.reasoning_effort,
                    chat_type=request.chat_type,
                    aspect_ratio=request.aspect_ratio,
                    use_single_prompt=use_single_prompt,
                    server_system_prompt=config.SERVER_SYSTEM_PROMPT
                ):
                    phase = raw_chunk.get("phase")
                    if not phase:
                        choices = raw_chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            phase = delta.get("phase")

                    content = raw_chunk.get("content", "")
                    if not content:
                        choices = raw_chunk.get("choices", [])
                        if choices:
                            content = choices[0].get("delta", {}).get("content", "")

                    if phase != "think":
                        full_content_buffer += content

                    openai_delta = {}
                    if phase == "think":
                        if content:
                            openai_delta["reasoning_content"] = content
                    else:
                        if content:
                            openai_delta["content"] = content

                    finish_reason = None
                    if "choices" in raw_chunk and raw_chunk["choices"]:
                        finish_reason = raw_chunk["choices"][0].get("finish_reason")

                    if openai_delta or finish_reason:
                        chunk_data = {
                            "id": conversation_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": openai_delta,
                                "finish_reason": finish_reason
                            }]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                    if finish_reason:
                        # Warn if final response too short
                        if len(full_content_buffer) < config.RETRY_MIN_LENGTH:
                            print(f"⚠️ Streaming response too short ({len(full_content_buffer)} chars). Consider retrying.")
                        break

                    if "usage" in raw_chunk:
                        usage_chunk = {
                            "id": conversation_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            "usage": raw_chunk["usage"]
                        }
                        yield f"data: {json.dumps(usage_chunk)}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                if config.DEBUG:
                    import traceback
                    traceback.print_exc()
                error_data = {"error": {"message": str(e), "type": "server_error"}}
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # NON-STREAMING MODE with automatic retry
        try:
            full_content, full_reasoning, usage, finish_reason = await QwenClient.complete_chat_with_retry(
                model=model,
                messages=messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=max_tokens,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                reasoning_effort=request.reasoning_effort,
                chat_type=request.chat_type,
                aspect_ratio=request.aspect_ratio,
                use_single_prompt=use_single_prompt,
                server_system_prompt=config.SERVER_SYSTEM_PROMPT
            )

            message = {"role": "assistant", "content": full_content}
            if full_reasoning:
                message["reasoning_content"] = full_reasoning

            return {
                "id": conversation_id,
                "object": "chat.completion",
                "created": int(time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason
                }],
                "usage": usage
            }
        except Exception as e:
            if config.DEBUG:
                import traceback
                traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def run_server():
    try:
        port = int(config.PORT)
    except (TypeError, ValueError):
        port = 8080

    # Print retry settings on startup
    print(f"🚀 Starting QwenProxy with retry: max_attempts={config.RETRY_MAX_ATTEMPTS}, min_length={config.RETRY_MIN_LENGTH}, delay={config.RETRY_DELAY}s")

    uvicorn.run(
        app,
        host=config.HOST,
        port=port,
        log_level="info" if not config.DEBUG else "debug",
        access_log=False
    )

if __name__ == "__main__":
    run_server()
