import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from signoff.core import flip_rate, matches, run_suite, sign_test, summarize

def test_matches():
    assert matches("abc", "abc") == (True, "exact")
    assert matches("Yes.", "yes") == (True, "normalized")
    assert matches("$1,000", "1000") == (True, "numeric")
    assert matches("abc", "xyz") == (False, "none")
    assert matches("abc", "xyz", judge=lambda a, b: True) == (True, "judge")

def test_flip_rate():
    assert flip_rate(["a"] * 5) == 0.0
    assert flip_rate(["a", "a", "b", "a", "a"]) == 0.2

def test_sign_test():
    assert sign_test(0, 0) == 1.0
    assert sign_test(10, 0) < 0.05
    assert 0 < sign_test(3, 2) <= 1.0

def test_run_suite_end_to_end():
    script = {"m:a": "42", "m:b": "forty-two"}
    def fake(spec, prompt, temperature):
        return script[spec]
    cases = [{"id": "q1", "prompt": "answer?", "expected": "42"}]
    results = run_suite(cases, "m:a", "m:b", runs=3, caller=fake)
    s = summarize(results, 3)
    assert results[0]["correct_a"] is True
    assert results[0]["correct_b"] is False
    assert results[0]["flip_a"] == 0.0
    assert s["accuracy_a"] == 1.0 and s["accuracy_b"] == 0.0
    assert s["verdict"] == "PASS"  # 1 discordant pair -> p=1.0, not significant

def test_stability_and_ablation():
    from signoff.core import stability_suite, stability_summary
    calls = {"n": 0}
    def flaky(spec, prompt, temperature):
        calls["n"] += 1
        if "Reply with only" not in prompt:   # rules section ablated: fully unstable
            return "X" if calls["n"] % 2 else "Y"
        return "A" if calls["n"] % 5 else "B"    # rules present: flips 1 in 5
    cases = [{"id": "c1", "sections": [
        {"name": "role", "text": "You extract dates."},
        {"name": "rules", "text": "Reply with only the date."}]}]
    results = stability_suite(cases, "m:x", runs=10, ablate=True, caller=flaky)
    r = results[0]
    assert r["flip_rate"] == 0.2 and r["distinct"] == 2
    assert set(r["ablation_flip_rates"]) == {"role", "rules"}
    assert r["ablation_flip_rates"]["rules"] == 0.5  # removing rules doubles instability
    s = stability_summary(results, 10, gate=0.1)
    assert s["verdict"] == "FAIL" and s["worst_flip"] == 0.2
    assert stability_summary(results, 10, gate=0.3)["verdict"] == "PASS"

def test_jev_judge_threshold():
    import io, json as j, urllib.request
    from signoff.core import jev_judge
    import os
    os.environ.setdefault("JEV_AI_API_KEY", "test-key-not-real")
    canned = j.dumps({"answers": {"same": {"noul": 0.8}}}).encode()
    class FakeResp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    real = urllib.request.urlopen
    urllib.request.urlopen = lambda req, timeout=0: FakeResp(canned)
    try:
        assert jev_judge(threshold=0.5)("a", "b") is True
        assert jev_judge(threshold=0.9)("a", "b") is False
    finally:
        urllib.request.urlopen = real

def test_stability_report_renders():
    from signoff.core import stability_suite, stability_summary
    from signoff.report import build_stability_html
    def steady(spec, prompt, temperature):
        return "same"
    results = stability_suite([{"prompt": "q"}], "m:x", runs=3, caller=steady)
    html = build_stability_html(stability_summary(results, 3), results, "m:x")
    assert "stability report" in html and "REPORT" in html

def test_report_renders():
    from signoff.report import build_html
    def fake(spec, prompt, temperature):
        return "same"
    results = run_suite([{"prompt": "q"}], "m:a", "m:b", runs=2, caller=fake)
    html = build_html(summarize(results, 2), results, "m:a", "m:b")
    assert "PASS" in html and "<table>" in html

if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
    print("all checks pass")
