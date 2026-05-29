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
    SINGLE_PROMPT_MODE = os.getenv("QWENPROXY_SINGLE_PROMPT", "false").lower() == "true"
    # NEW: if True, sends full conversation history; if False, only last user message
    FULL_HISTORY = os.getenv("QWENPROXY_FULL_HISTORY", "true").lower() == "true"
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

# ==================== Helper Functions ====================

def build_conversation_prompt(messages: list[ChatMessage]) -> str:
    """Build a full conversation string for single-prompt mode."""
    lines = []
    for msg in messages:
        role = "SYSTEM" if msg.role == "system" else "AI" if msg.role == "assistant" else "USER"
        lines.append(f"{role}: {msg.get_content_str()}")
    return "\n".join(lines)

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

    @staticmethod
    def _extract_content_from_chunk(chunk: dict) -> tuple[str, str]:
        """Extract (content, reasoning_content) from a raw chunk."""
        content = ""
        reasoning = ""

        # Direct fields
        if "content" in chunk:
            content = chunk.get("content", "")
        if "reasoning_content" in chunk:
            reasoning = chunk.get("reasoning_content", "")

        # Choices -> delta
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            if not content:
                content = delta.get("content", "")
            if not reasoning:
                reasoning = delta.get("reasoning_content", "")
            if not content and "text" in delta:
                content = delta["text"]

        # Phase separation
        if not content and not reasoning:
            phase = chunk.get("phase")
            text = chunk.get("text", "")
            if phase == "think":
                reasoning = text
            else:
                content = text

        return content, reasoning

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
        server_system_prompt: str = "",
        stop: Optional[list[str]] = None
    ) -> AsyncGenerator[dict, None]:
        """Streams raw Qwen API chunks."""
        enable_thinking = reasoning_effort in ("medium", "high")
        thinking_mode = "Auto" if enable_thinking else "Fast"

        # Build the user prompt
        if use_single_prompt or not config.FULL_HISTORY:
            merged = merge_system_prompts(messages, server_system_prompt)
            prompt = build_conversation_prompt(merged)
        else:
            # Send full conversation history as messages array (not just last user)
            # Qwen's API expects a 'messages' array with role/content.
            # We'll construct that instead of a single prompt.
            qwen_messages = []
            for msg in messages:
                if msg.role == "system" and server_system_prompt:
                    # Merge system prompts
                    qwen_messages.append({
                        "role": "system",
                        "content": f"{server_system_prompt}\n\n{msg.get_content_str()}"
                    })
                else:
                    qwen_messages.append({
                        "role": msg.role,
                        "content": msg.get_content_str()
                    })
            # We'll use a special flag to indicate that we're sending a full messages array
            use_messages_array = True
        if use_single_prompt or not config.FULL_HISTORY:
            use_messages_array = False
            prompt = build_conversation_prompt(merge_system_prompts(messages, server_system_prompt))

        message_id = str(uuid.uuid4())

        async with aiohttp.ClientSession(headers=cls._get_headers()) as session:
            req_headers = await cls._get_req_headers(session)

            # Create a new chat
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
                    raise RuntimeError(f"Failed to create chat: {resp.status}")
                data = await resp.json()
                if not (data.get('success') and data.get('data', {}).get('id')):
                    raise RuntimeError(f"Failed to create chat: {data}")
                chat_id = data['data']['id']

            # Prepare feature config
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

            # Build the message payload with proper generation parameters
            # According to Qwen's API, parameters like max_tokens, stop, etc. go inside a 'generation_config' object
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
            }
            if stop:
                generation_config["stop"] = stop

            msg_payload = {
                "stream": True,
                "incremental_output": True,
                "chat_id": chat_id,
                "chat_mode": "normal",
                "model": model,
                "parent_id": None,
                "messages": [] if use_messages_array else [{
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
                    "generation_config": generation_config
                }]
            }
            if use_messages_array:
                # When sending full conversation history, the structure is different
                # We need to embed the messages array directly
                msg_payload["messages"] = [
                    {
                        "fid": str(uuid.uuid4()),
                        "parentId": None,
                        "childrenIds": [],
                        "role": m["role"],
                        "content": m["content"],
                        "user_action": "chat" if m["role"] == "user" else None,
                        "files": [],
                        "models": [model],
                        "chat_type": chat_type,
                        "feature_config": feature_config,
                        "sub_chat_type": chat_type,
                        "generation_config": generation_config
                    } for m in qwen_messages
                ]
                # The last message should have user_action = "chat"
                if msg_payload["messages"]:
                    msg_payload["messages"][-1]["user_action"] = "chat"

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
                                print(f"[RAW] {json.dumps(parsed, ensure_ascii=False)[:300]}")
                            yield parsed
                        except json.JSONDecodeError:
                            if config.DEBUG:
                                print(f"[SKIP] {data_str[:100]}")
                            continue
                if buffer.strip():
                    try:
                        parsed = json.loads(buffer)
                        if config.DEBUG:
                            print(f"[RAW final] {json.dumps(parsed, ensure_ascii=False)[:300]}")
                        yield parsed
                    except:
                        pass

    # Non‑streaming with retry
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
        server_system_prompt: str = "",
        stop: Optional[list[str]] = None
    ) -> tuple[str, str, Optional[dict], str]:
        attempt = 1
        last_error = None
        retry_temperature = temperature
        retry_reasoning = reasoning_effort

        while attempt <= config.RETRY_MAX_ATTEMPTS:
            full_content = ""
            full_reasoning = ""
            usage = None
            finish_reason = "stop"

            try:
                async for raw_chunk in cls.create_chat(
                    model=model,
                    messages=messages,
                    temperature=retry_temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    reasoning_effort=retry_reasoning,
                    chat_type=chat_type,
                    aspect_ratio=aspect_ratio,
                    use_single_prompt=use_single_prompt,
                    server_system_prompt=server_system_prompt,
                    stop=stop
                ):
                    c, r = cls._extract_content_from_chunk(raw_chunk)
                    full_content += c
                    full_reasoning += r
                    if "choices" in raw_chunk and raw_chunk["choices"]:
                        fr = raw_chunk["choices"][0].get("finish_reason")
                        if fr:
                            finish_reason = fr
                    if "usage" in raw_chunk:
                        usage = raw_chunk["usage"]

                total_len = len(full_content) + len(full_reasoning)
                if total_len >= config.RETRY_MIN_LENGTH:
                    return full_content, full_reasoning, usage, finish_reason
                else:
                    if config.DEBUG:
                        print(f"⚠️ Short response ({total_len} chars), retry {attempt}/{config.RETRY_MAX_ATTEMPTS}")
                    retry_temperature = max(0.1, retry_temperature * 0.7)
                    retry_reasoning = "low"
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

app = FastAPI(title="QwenProxy", version="3.3.0")
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
    model = request.model
    if model not in all_models:
        raise HTTPException(400, f"Model '{model}' not found")

    conv_id = str(uuid.uuid4())
    use_single = request.use_single_prompt if request.use_single_prompt is not None else config.SINGLE_PROMPT_MODE
    max_tokens = request.max_tokens or 4096

    if request.stream:
        async def stream_generator():
            created = int(time.time())
            full_content_len = 0
            finished_normally = False
            try:
                async for raw in QwenClient.create_chat(
                    model=model,
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=max_tokens,
                    presence_penalty=request.presence_penalty,
                    frequency_penalty=request.frequency_penalty,
                    reasoning_effort=request.reasoning_effort,
                    chat_type=request.chat_type,
                    aspect_ratio=request.aspect_ratio,
                    use_single_prompt=use_single,
                    server_system_prompt=config.SERVER_SYSTEM_PROMPT,
                    stop=request.stop
                ):
                    content, reasoning = QwenClient._extract_content_from_chunk(raw)
                    full_content_len += len(content)
                    finish_reason = None
                    if "choices" in raw and raw["choices"]:
                        finish_reason = raw["choices"][0].get("finish_reason")

                    delta = {}
                    if reasoning:
                        delta["reasoning_content"] = reasoning
                    if content:
                        delta["content"] = content

                    # Do not send finish_reason if we have very little content
                    if finish_reason and full_content_len < config.RETRY_MIN_LENGTH:
                        if config.DEBUG:
                            print(f"Ignoring premature finish_reason ({finish_reason}) after {full_content_len} chars")
                        finish_reason = None

                    if delta or finish_reason:
                        chunk = {
                            "id": conv_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    if finish_reason:
                        finished_normally = True
                        break

                    if "usage" in raw:
                        yield f"data: {json.dumps({'id': conv_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': raw['usage']})}\n\n"
                        finished_normally = True
                        break

                if not finished_normally and full_content_len > 0:
                    # Send final stop chunk
                    yield f"data: {json.dumps({'id': conv_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                if config.DEBUG:
                    import traceback
                    traceback.print_exc()
                yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
                yield "data: [DONE]\n\n"
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        # Non‑streaming with retry
        try:
            content, reasoning, usage, finish_reason = await QwenClient.complete_chat_with_retry(
                model=model,
                messages=request.messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=max_tokens,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                reasoning_effort=request.reasoning_effort,
                chat_type=request.chat_type,
                aspect_ratio=request.aspect_ratio,
                use_single_prompt=use_single,
                server_system_prompt=config.SERVER_SYSTEM_PROMPT,
                stop=request.stop
            )
            msg = {"role": "assistant", "content": content}
            if reasoning:
                msg["reasoning_content"] = reasoning
            return {
                "id": conv_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
                "usage": usage or {}
            }
        except Exception as e:
            raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

def run_server():
    print(f"🚀 QwenProxy v3.3.0 on {config.HOST}:{config.PORT}")
    print(f"   Full conversation history: {config.FULL_HISTORY}")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")

if __name__ == "__main__":
    run_server()
