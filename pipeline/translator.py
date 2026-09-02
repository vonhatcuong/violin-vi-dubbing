"""Translate transcript segments via configurable LLM provider (OpenAI or Together AI)."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from together import (
    APITimeoutError as TogetherTimeout,
    InternalServerError as TogetherISE,
    RateLimitError as TogetherRateLimit,
)
from openai import (
    APITimeoutError as OpenAITimeout,
    InternalServerError as OpenAIISE,
    RateLimitError as OpenAIRateLimit,
)

try:
    from openai import APIConnectionError as OpenAIConn
except ImportError:
    OpenAIConn = None
try:
    import httpx as _httpx
    _HTTPX_ERRS: tuple = (_httpx.ReadError, _httpx.ConnectError, _httpx.RemoteProtocolError)
except ImportError:
    _HTTPX_ERRS = ()

_TRANSIENT_ERRORS: tuple = (
    TogetherTimeout, TogetherISE, TogetherRateLimit,
    OpenAITimeout, OpenAIISE, OpenAIRateLimit,
    *((OpenAIConn,) if OpenAIConn else ()),
    *_HTTPX_ERRS,
)

from . import config as _conf
from .costs import CostTracker
from .llm_client import get_translation_model, get_translation_provider
from .transcriber import Segment

import prompts as _prompts


def _tcfg() -> dict:
    return _conf.get()["translation"]


def _asr_corrections_block() -> str:
    corrections = _tcfg().get("asr_corrections") or []
    if not corrections:
        return ""
    lines = "\n".join(f"  - {c}" for c in corrections)
    return (
        "\nPre-translation text fixes (apply each as a literal find-and-replace "
        "on the SOURCE before translating; do not include the left-side form "
        "in your output):\n"
        f"{lines}\n"
    )

BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}

SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"},
    },
    "required": ["translation"],
    "additionalProperties": False,
}


def _response_format(name: str, schema: dict) -> dict:
    """Build the `response_format` kwarg per `translation.response_format`.

    json_schema (default) — strict schema; supported by OpenAI/Together.
    json_object            — plain JSON mode; safer for Ollama/vLLM, which have
                             hung on strict json_schema (see commit 3957df5).
    """
    mode = _tcfg().get("response_format", "json_schema")
    if mode == "json_object":
        return {"type": "json_object"}
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


def _is_local_provider(cfg: dict) -> bool:
    """Local LLM servers (Ollama, vLLM/llama.cpp via openai_compat) get the /no_think switch."""
    return get_translation_provider(cfg) in ("ollama", "openai_compat")


def _provider_extra() -> dict[str, Any]:
    """Provider-specific `extra_body` kwargs that switch off hidden reasoning.

    together      → chat_template_kwargs.enable_thinking=False (Qwen on Together).
    ollama        → reasoning_effort (default "none"): Gemma 4 thinks by default and
                    the OpenAI-compatible endpoint ignores `think: false`; measured
                    13 s → 1.1 s per sentence on gemma4:31b.
    openai_compat → reasoning_effort only when `translation.reasoning_effort` is set
                    (some servers reject unknown fields).
    """
    cfg = _conf.get()
    provider = get_translation_provider(cfg)
    if provider == "together":
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    effort = _tcfg().get("reasoning_effort", "none" if provider == "ollama" else None)
    if provider in ("ollama", "openai_compat") and effort:
        return {"extra_body": {"reasoning_effort": effort}}
    return {}


def _translate_single(
    text: str,
    target_language: str,
    source_language: str,
    client: Any,
    tracker: CostTracker | None = None,
    style_directives: str = "",
    style_temperature: float | None = None,
) -> str:
    """Translate one segment with retry on transient API errors."""
    cfg = _conf.get()
    model = get_translation_model(cfg)
    max_retries = cfg["translation"]["max_retries"]
    temp = style_temperature if style_temperature is not None else cfg["translation"]["temperature"]

    fmt = dict(
        source_language=source_language,
        target_language=target_language,
        text=json.dumps(text, ensure_ascii=False),
        style_directives=style_directives,
        asr_corrections_block=_asr_corrections_block(),
    )
    if style_directives:
        system_msg = _prompts.load("translate", "single_system_styled", **fmt)
        user_msg = _prompts.load("translate", "single_user_styled", **fmt)
    else:
        system_msg = _prompts.load("translate", "single_system", **fmt)
        user_msg = _prompts.load("translate", "single_user", **fmt)

    # qwen3 (Ollama) reasons by default, making each call minutes-slow; the
    # /no_think soft switch disables it. Together uses extra_body instead (below).
    if _is_local_provider(cfg):
        system_msg = "/no_think\n" + system_msg

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temp,
                response_format=_response_format("single_translation", SINGLE_SCHEMA),
                **_provider_extra(),
            )
            if tracker and hasattr(response, "usage") and response.usage:
                tracker.add_llm_usage(
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0,
                )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)["translation"]

        except _TRANSIENT_ERRORS as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"        ⚠ API error (attempt {attempt}): {exc}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def _try_batch(
    texts: list[str],
    target_language: str,
    source_language: str,
    client: Any,
    tracker: CostTracker | None = None,
    style_directives: str = "",
    style_temperature: float | None = None,
    budgets: list[tuple[float, int]] | None = None,
) -> list[str] | None:
    """Attempt to translate a batch. Returns translations on success, None on failure."""
    if budgets:
        numbered = "\n".join(
            f"[{i}] ({sec:.1f}s, ≤{syl} syllables): {json.dumps(t, ensure_ascii=False)}"
            for i, (t, (sec, syl)) in enumerate(zip(texts, budgets))
        )
        length_block = _prompts.load("translate", "budget_block")
    else:
        numbered = "\n".join(f"[{i}]: {json.dumps(t, ensure_ascii=False)}" for i, t in enumerate(texts))
        length_block = _prompts.load("translate", "length_block_free", target_language=target_language)

    fmt = dict(
        source_language=source_language,
        target_language=target_language,
        num_segments=len(texts),
        numbered_segments=numbered,
        style_directives=style_directives,
        asr_corrections_block=_asr_corrections_block(),
        length_block=length_block,
    )
    if style_directives:
        system_msg = _prompts.load("translate", "batch_system_styled", **fmt)
        prompt = _prompts.load("translate", "batch_user_styled", **fmt)
    else:
        system_msg = _prompts.load("translate", "batch_system", **fmt)
        prompt = _prompts.load("translate", "batch_user", **fmt)

    cfg = _conf.get()
    if _is_local_provider(cfg):
        system_msg = "/no_think\n" + system_msg
    model = get_translation_model(cfg)
    max_retries = cfg["translation"]["max_retries"]
    temp = style_temperature if style_temperature is not None else cfg["translation"]["temperature"]
    for attempt in range(1, max_retries + 1):
        raw = ""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                response_format=_response_format("translation_response", BATCH_SCHEMA),
                **_provider_extra(),
            )

            if tracker and hasattr(response, "usage") and response.usage:
                tracker.add_llm_usage(
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0,
                )

            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            translated = result["translations"]

            if len(translated) == len(texts):
                return translated

            if attempt < max_retries:
                print(f"      ⚠ Count mismatch (attempt {attempt}): expected {len(texts)}, got {len(translated)}, retrying...")
                time.sleep(2 ** attempt)

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            if attempt < max_retries:
                print(f"      ⚠ Parse error (attempt {attempt}): {exc}, retrying...")
                time.sleep(2 ** attempt)

        except _TRANSIENT_ERRORS as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"      ⚠ API error (attempt {attempt}): {exc}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"      ✗ API error after {max_retries} attempts: {exc}")

    return None


def _translate_batch(
    texts: list[str],
    target_language: str,
    source_language: str,
    client: Any,
    tracker: CostTracker | None = None,
    style_directives: str = "",
    style_temperature: float | None = None,
    budgets: list[tuple[float, int]] | None = None,
) -> list[str]:
    """Translate a batch with binary-split fallback on failure."""
    result = _try_batch(texts, target_language, source_language, client, tracker, style_directives, style_temperature, budgets)
    if result is not None:
        return result

    if len(texts) == 1:
        print(f"        → single-segment fallback...", end="", flush=True)
        t = _translate_single(texts[0], target_language, source_language, client, tracker, style_directives, style_temperature)
        print(" done")
        return [t]

    mid = len(texts) // 2
    print(f"      ↓ Splitting failed batch of {len(texts)} → {mid} + {len(texts) - mid}")
    left_budgets = budgets[:mid] if budgets else None
    right_budgets = budgets[mid:] if budgets else None
    left = _translate_batch(texts[:mid], target_language, source_language, client, tracker, style_directives, style_temperature, left_budgets)
    right = _translate_batch(texts[mid:], target_language, source_language, client, tracker, style_directives, style_temperature, right_budgets)
    return left + right


def translate_segments(
    segments: list[Segment],
    target_language: str,
    client: Any,
    source_language: str = "auto-detect",
    tracker: CostTracker | None = None,
    style_directives: str = "",
    style_temperature: float | None = None,
    budgets: list[tuple[float, int]] | None = None,
) -> list[Segment]:
    """Translate all segments, batching to stay within LLM context limits."""
    tcfg = _tcfg()
    batch_size = tcfg["batch_size"]
    parallel_batches = tcfg.get("parallel_batches", 1)

    batches = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        texts = [s.text for s in batch]
        batch_budgets = budgets[i : i + batch_size] if budgets else None
        batches.append((i, texts, batch_budgets))

    translated_texts: list[str] = [""] * len(segments)

    if parallel_batches <= 1:
        for i, texts, batch_budgets in batches:
            print(f"      Translating segments {i + 1}–{i + len(texts)} / {len(segments)}...")
            translated_texts[i : i + len(texts)] = _translate_batch(
                texts, target_language, source_language, client, tracker,
                style_directives, style_temperature, batch_budgets,
            )
    else:
        with ThreadPoolExecutor(max_workers=parallel_batches) as executor:
            future_to_batch = {
                executor.submit(
                    _translate_batch, texts, target_language, source_language, client,
                    tracker, style_directives, style_temperature, batch_budgets,
                ): (i, texts)
                for i, texts, batch_budgets in batches
            }
            for future in as_completed(future_to_batch):
                i, texts = future_to_batch[future]
                translated_texts[i : i + len(texts)] = future.result()
                print(f"      Translated segments {i + 1}–{i + len(texts)} / {len(segments)}...")

    return [
        Segment(
            id=s.id, start=s.start, end=s.end, text=t,
            speaker=s.speaker, source_text=s.text,
        )
        for s, t in zip(segments, translated_texts)
    ]


def shorten_segment(
    source_text: str,
    current_text: str,
    budget_syllables: int,
    budget_seconds: float,
    target_language: str,
    client: Any,
    tracker: CostTracker | None = None,
    source_language: str = "English",
) -> str:
    """Ask the LLM for a shorter translation that fits `budget_syllables`.

    Returns `current_text` unchanged when the model fails, so the fitter can
    always continue (speed-up + merger will absorb the overrun instead).
    """
    from .vi_text import count_syllables

    cfg = _conf.get()
    fmt = dict(
        source_language=source_language,
        target_language=target_language,
        source_text=source_text,
        current_text=current_text,
        current_syllables=count_syllables(current_text),
        budget_syllables=budget_syllables,
        budget_seconds=f"{budget_seconds:.1f}",
    )
    system_msg = _prompts.load("translate", "shorten_system", **fmt)
    user_msg = _prompts.load("translate", "shorten_user", **fmt)
    if _is_local_provider(cfg):
        system_msg = "/no_think\n" + system_msg
    try:
        response = client.chat.completions.create(
            model=get_translation_model(cfg),
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.2,
            response_format=_response_format("shortened_translation", SINGLE_SCHEMA),
            **_provider_extra(),
        )
        if tracker and getattr(response, "usage", None):
            tracker.add_llm_usage(response.usage.prompt_tokens or 0, response.usage.completion_tokens or 0)
        text = json.loads(response.choices[0].message.content.strip())["translation"].strip()
        return text or current_text
    except Exception as exc:  # JSON errors, API errors — never break the pipeline
        print(f"        ⚠ shorten failed ({exc}); keeping current text")
        return current_text
