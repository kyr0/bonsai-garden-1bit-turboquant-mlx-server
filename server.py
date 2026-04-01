"""OpenAI-compatible FastAPI server for bonsai - powered by mlx_lm."""

import time
import traceback
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from mlx_lm import load, generate

logger = logging.getLogger("bonsai")

MODEL_NAME = "prism-ml/Bonsai-8B-mlx-1bit"
model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    logger.info("Loading model %s ...", MODEL_NAME)
    model, tokenizer = load(MODEL_NAME)
    logger.info("Model loaded. Running warmup ...")
    _ = generate(model, tokenizer, prompt="Hello", max_tokens=2)
    logger.info("Warmup complete.")
    yield


app = FastAPI(title="bonsai", lifespan=lifespan)

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[Message]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

class CompletionRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

def _generate(prompt: str, max_tokens: int) -> tuple[str, int, int, float]:
    """Return (text, prompt_tokens, completion_tokens, gen_seconds)."""
    prompt_toks = tokenizer.encode(prompt)
    prompt_tokens = len(prompt_toks)
    t0 = time.perf_counter()
    text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    gen_seconds = time.perf_counter() - t0
    completion_tokens = len(tokenizer.encode(text))
    return text, prompt_tokens, completion_tokens, gen_seconds


def _build_chat_prompt(messages: list[Message]) -> str:
    """Simple chat prompt builder."""
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [m.model_dump() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(f"System: {m.content}\n")
        elif m.role == "user":
            parts.append(f"User: {m.content}\n")
        elif m.role == "assistant":
            parts.append(f"Assistant: {m.content}\n")
    parts.append("Assistant:")
    return "".join(parts)


def _ts() -> int:
    return int(time.time())

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "ready": model is not None}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": _ts(),
                "owned_by": "bonsai",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        prompt = _build_chat_prompt(req.messages)
        logger.info("Chat prompt length: %d chars", len(prompt))
        text, prompt_tokens, completion_tokens, gen_s = _generate(prompt, req.max_tokens)
        tok_s = round(completion_tokens / gen_s, 1) if gen_s > 0 else 0
        logger.info("Chat: %d prompt tok, %d compl tok, %.2fs, %.1f tok/s",
                    prompt_tokens, completion_tokens, gen_s, tok_s)
    except Exception:
        logger.error("Chat completion failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": _ts(),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "timing": {
            "generation_seconds": round(gen_s, 3),
            "tokens_per_second": tok_s,
        },
    }


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        logger.info("Completion prompt length: %d chars", len(req.prompt))
        text, prompt_tokens, completion_tokens, gen_s = _generate(req.prompt, req.max_tokens)
        tok_s = round(completion_tokens / gen_s, 1) if gen_s > 0 else 0
        logger.info("Completion: %d prompt tok, %d compl tok, %.2fs, %.1f tok/s",
                    prompt_tokens, completion_tokens, gen_s, tok_s)
    except Exception:
        logger.error("Completion failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    resp_id = f"cmpl-{uuid.uuid4().hex[:8]}"

    return {
        "id": resp_id,
        "object": "text_completion",
        "created": _ts(),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "timing": {
            "generation_seconds": round(gen_s, 3),
            "tokens_per_second": tok_s,
        },
    }
