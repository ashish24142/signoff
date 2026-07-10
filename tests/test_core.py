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
