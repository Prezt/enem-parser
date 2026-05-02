#!/usr/bin/env python3
"""
Local Anthropic API proxy that routes requests through the `claude` CLI.

Allows enem-parser to use Claude via your Claude subscription instead of
direct API credits.

Usage:
    python claude_proxy.py          # starts on http://localhost:8888

Then in .env set:
    ANTHROPIC_BASE_URL=http://localhost:8888

The Anthropic SDK automatically picks up ANTHROPIC_BASE_URL, so no other
code changes are needed.
"""

import subprocess
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if block.get("type") == "text"
        )
    return ""


@app.post("/v1/messages")
async def messages(request: Request) -> JSONResponse:
    body = await request.json()

    msgs = body.get("messages", [])
    model = body.get("model", "claude-sonnet-4-6")
    max_tokens = body.get("max_tokens", 4096)

    # Detect assistant-turn prefill (last message with role="assistant")
    prefill = ""
    if msgs and msgs[-1]["role"] == "assistant":
        prefill = _extract_text(msgs[-1]["content"])
        msgs = msgs[:-1]

    # Build prompt — concatenate user turns; skip prior assistant turns
    parts = []
    for msg in msgs:
        if msg["role"] == "user":
            parts.append(_extract_text(msg["content"]))

    prompt = "\n\n".join(parts)

    # Tell Claude to start with the prefill character so the caller's
    # `"{" + continuation` logic still works correctly.
    if prefill:
        prompt += (
            f"\n\nIMPORTANT: Your response MUST start IMMEDIATELY with the character "
            f"{prefill!r} — no preamble, no explanation, no markdown fences."
        )

    try:
        result = subprocess.run(
            ["claude", "--model", model, "-p"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {"type": "server_error", "message": "claude CLI timed out"},
            },
        )

    if result.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "server_error",
                    "message": result.stderr or "claude CLI error",
                },
            },
        )

    text = result.stdout.strip()

    # Strip the prefill prefix so the caller gets only the continuation.
    # The AnthropicClient does `"{" + continuation`, so we must not include
    # the leading `{` in what we return.
    if prefill and text.startswith(prefill):
        text = text[len(prefill):]

    return JSONResponse(
        content={
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    )


@app.get("/")
async def health():
    return {"status": "ok", "proxy": "claude-cli"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8888, log_level="info")
