# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Build the program to run from a chat model's reply, for the `humaneval_chat` and `mbpp_chat` tasks.

lm-evaluation-harness loads this file by path from the task directory (`!function code_chat_utils.build_humaneval`), so it imports nothing from
this repository and nothing outside the standard library until a score is computed. The tests load it the same way.

A chat reply is a few sentences and a fenced code block. The block may hold the whole function, the whole prompt and the body, or only the body;
there may be a second block with example calls. `choose_block` picks the block that defines the function, and `humaneval_program` puts the
problem's own prompt in front of it, so helper functions and imports in the prompt stay available and the reply's definition replaces the
prompt's stub. Module-level `if __name__ == ...` code is cut, because example calls there can fail a run for a reason that has nothing to do with
the function.
"""
from __future__ import annotations

import re

PYTHON_LANGS = ("", "python", "py", "python3")
FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)
MAIN_GUARD = re.compile(r"^if\s+__name__\s*==", re.MULTILINE)


def extract_blocks(text: str) -> list[tuple[str, str]]:
    """Each fenced block as (language, code). A reply cut off inside a block keeps what it wrote."""
    return [(lang, code) for lang, code in FENCE.findall(text or "")]


def defines(code: str, name: str) -> bool:
    return re.search(rf"^[ \t]*(?:async\s+)?def\s+{re.escape(name)}\s*\(", code, re.MULTILINE) is not None


def choose_block(blocks: list[tuple[str, str]], entry_point):
    """The code to use: a Python (or unlabeled) block that defines the entry point, else one that defines any function, else the first Python
    block, else the first block of any language. None when there are no blocks."""
    if not blocks:
        return None
    py = [code for lang, code in blocks if lang.lower() in PYTHON_LANGS]
    if entry_point:
        for code in py:
            if defines(code, entry_point):
                return code
    for code in py:
        if re.search(r"^[ \t]*def\s", code, re.MULTILINE):
            return code
    return py[0] if py else blocks[0][1]


def cut_main_guard(code: str) -> str:
    return MAIN_GUARD.split(code, maxsplit=1)[0]


def _indent_body(code: str) -> str:
    lines = code.splitlines()
    first = next((l for l in lines if l.strip()), "")
    if first[:1] in (" ", "\t"):
        return code if code.endswith("\n") else code + "\n"
    return "".join(("    " + l if l.strip() else l) + "\n" for l in lines)


def humaneval_program(reply: str, doc: dict) -> str:
    entry, prompt = doc["entry_point"], doc["prompt"]
    code = choose_block(extract_blocks(reply), entry)
    if code is None:
        code = reply or ""
    code = cut_main_guard(code)
    if not code.strip():
        return prompt                                  # the stub the prompt ends with: a docstring body that returns None
    if defines(code, entry):
        return prompt + "\n\n" + code.rstrip() + "\n"
    return prompt + _indent_body(code.rstrip("\n"))


def mbpp_program(reply: str, doc: dict) -> str:
    code = choose_block(extract_blocks(reply), None)
    if code is None:
        code = reply or ""
    code = cut_main_guard(code).rstrip() + "\n" if code.strip() else ""
    setup = (doc.get("test_setup_code") or "").strip()
    return (setup + "\n" if setup else "") + code


def build_humaneval(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[humaneval_program(r, doc) for r in resp] for resp, doc in zip(resps, docs)]


def build_mbpp(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[mbpp_program(r, doc) for r in resp] for resp, doc in zip(resps, docs)]


# ---- scoring: the same `code_eval` metric the harness's own code tasks use, loaded when first needed ----------------------------------------

_code_eval = None


def _compute():
    global _code_eval
    if _code_eval is None:
        import evaluate
        _code_eval = evaluate.load("code_eval")
    return _code_eval


def pass_at_k(references, predictions, k=None):
    assert k is not None
    if isinstance(k, int):
        k = [k]
    return _compute().compute(references=references, predictions=predictions, k=k)[0]


def pass_at_1(references, predictions):
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    return _compute().compute(references=references, predictions=predictions, k=[1])[0]["pass@1"]
