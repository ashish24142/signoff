"""signoff core: provider adapters, layered matching, run engine, stats."""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

# ---------- providers ----------

@lru_cache(maxsize=None)
def _openai_client():
    from openai import OpenAI
    return OpenAI()

@lru_cache(maxsize=None)
def _anthropic_client():
    import anthropic
    return anthropic.Anthropic()

def _openai(model: str, prompt: str, temperature: float) -> str:
    r = _openai_client().chat.completions.create(
        model=model, temperature=temperature,
        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content or ""

def _anthropic(model: str, prompt: str, temperature: float) -> str:
    r = _anthropic_client().messages.create(
        model=model, max_tokens=1024, temperature=temperature,
        messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if b.type == "text")

PROVIDERS = {"openai": _openai, "anthropic": _anthropic}

def call(spec: str, prompt: str, temperature: float = 0.0) -> str:
    """spec is 'provider:model', e.g. 'openai:gpt-4o' or 'anthropic:claude-sonnet-4-5'."""
    provider, _, model = spec.partition(":")
    if provider not in PROVIDERS or not model:
        raise ValueError(f"bad model spec {spec!r}; use provider:model with provider in {sorted(PROVIDERS)}")
    return PROVIDERS[provider](model, prompt, temperature)

# ---------- layered matching ----------

_PUNCT = re.compile(r"[\s.,;:!?'\"`]+")

def _norm(s: str) -> str:
    return _PUNCT.sub(" ", s.strip().lower()).strip()

def _num(s: str):
    try:
        return float(s.strip().lstrip("$").rstrip("%").replace(",", ""))
    except ValueError:
        return None

def matches(a: str, b: str, judge=None) -> tuple[bool, str]:
    """Layered comparison -> (matched, layer). Layers: exact, normalized, numeric, judge."""
    if a == b:
        return True, "exact"
    if _norm(a) == _norm(b):
        return True, "normalized"
    na, nb = _num(a), _num(b)
    if na is not None and na == nb:
        return True, "numeric"
    if judge is not None and judge(a, b):
        return True, "judge"
    return False, "none"

def llm_judge(spec: str):
    """Judge factory: returns judge(a, b) -> bool using the given provider:model."""
    def judge(a: str, b: str) -> bool:
        verdict = call(spec, (
            "Do these two answers convey the same information? "
            "Reply with only YES or NO.\n\nAnswer A:\n{}\n\nAnswer B:\n{}".format(a, b)))
        return verdict.strip().upper().startswith("YES")
    return judge

# ---------- stats ----------

def majority(answers: list[str]) -> str:
    return Counter(answers).most_common(1)[0][0]

def flip_rate(answers: list[str]) -> float:
    """Fraction of runs disagreeing with the majority answer."""
    if len(answers) <= 1:
        return 0.0
    return (len(answers) - Counter(answers).most_common(1)[0][1]) / len(answers)

def sign_test(a_wins: int, b_wins: int) -> float:
    """Exact two-sided binomial sign test on discordant pairs; 1.0 when none."""
    n = a_wins + b_wins
    if n == 0:
        return 1.0
    k = min(a_wins, b_wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)

# ---------- run engine ----------

def run_suite(cases, model_a, model_b, runs=5, temperature=0.0, judge=None, caller=call):
    """cases: [{'id'?, 'prompt', 'expected'?}] -> per-case result rows."""
    results = []
    for case in cases:
        row = {"id": str(case.get("id") or case["prompt"][:40]), "prompt": case["prompt"]}
        for tag, spec in (("a", model_a), ("b", model_b)):
            # ponytail: sequential calls; ThreadPoolExecutor when suites get slow
            answers = [caller(spec, case["prompt"], temperature) for _ in range(runs)]
            row[f"answers_{tag}"] = answers
            row[f"majority_{tag}"] = majority(answers)
            row[f"flip_{tag}"] = flip_rate(answers)
        row["ab_match"], row["ab_layer"] = matches(row["majority_a"], row["majority_b"], judge)
        if "expected" in case:
            expected = str(case["expected"])
            row["expected"] = expected
            row["correct_a"] = matches(row["majority_a"], expected, judge)[0]
            row["correct_b"] = matches(row["majority_b"], expected, judge)[0]
        results.append(row)
    return results

def summarize(results, runs):
    n = len(results)
    s = {
        "cases": n, "runs": runs,
        "agreement": sum(r["ab_match"] for r in results) / n,
        "flip_a": sum(r["flip_a"] for r in results) / n,
        "flip_b": sum(r["flip_b"] for r in results) / n,
    }
    scored = [r for r in results if "correct_a" in r]
    if scored:
        s["accuracy_a"] = sum(r["correct_a"] for r in scored) / len(scored)
        s["accuracy_b"] = sum(r["correct_b"] for r in scored) / len(scored)
        a_wins = sum(r["correct_a"] and not r["correct_b"] for r in scored)
        b_wins = sum(r["correct_b"] and not r["correct_a"] for r in scored)
        s["a_wins"], s["b_wins"] = a_wins, b_wins
        s["p_value"] = sign_test(a_wins, b_wins)
        # fail only when B is worse AND the regression is statistically significant
        s["verdict"] = "PASS" if s["accuracy_b"] >= s["accuracy_a"] or s["p_value"] > 0.05 else "FAIL"
    else:
        # ponytail: no ground truth -> agreement heuristic; equivalence bands later
        s["verdict"] = "PASS" if s["agreement"] >= 0.9 else "REVIEW"
    return s
