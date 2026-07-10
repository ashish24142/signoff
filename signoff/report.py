"""HTML report — stdlib only."""
from html import escape

_PAGE = """<!doctype html><meta charset="utf-8"><title>signoff report</title>
<style>
body{{font-family:Segoe UI,system-ui,sans-serif;margin:40px auto;max-width:1080px;color:#1a2333;padding:0 16px}}
h1{{font-size:22px}} .verdict{{display:inline-block;padding:4px 14px;border-radius:4px;color:#fff;font-weight:600}}
.PASS{{background:#2e9e5b}} .FAIL{{background:#c0392b}} .REVIEW{{background:#d9912c}}
table{{border-collapse:collapse;width:100%;margin-top:18px;font-size:14px}}
th,td{{border:1px solid #d7dde7;padding:7px 10px;text-align:left;vertical-align:top}}
th{{background:#f0f3f8}} td.no{{background:#fdecea}} .mut{{color:#6b7787;font-size:13px}}
</style>
<h1>signoff &mdash; migration report <span class="verdict {verdict}">{verdict}</span></h1>
<p class="mut">model A: <b>{model_a}</b> &nbsp;vs&nbsp; model B: <b>{model_b}</b> &nbsp;&middot;&nbsp; {cases} cases &times; {runs} runs each</p>
<p>{summary_line}</p>
<table>
<tr><th>case</th><th>majority A</th><th>flip A</th><th>majority B</th><th>flip B</th><th>A=B (layer)</th><th>expected</th><th>A ok</th><th>B ok</th></tr>
{rows}
</table>
"""

def _cell_ok(v):
    if v is None:
        return "<td></td>"
    return '<td>yes</td>' if v else '<td class="no">no</td>'

def build_html(summary, results, model_a, model_b):
    line = "A/B agreement {:.1%} &middot; mean flip-rate A {:.1%}, B {:.1%}".format(
        summary["agreement"], summary["flip_a"], summary["flip_b"])
    if "accuracy_a" in summary:
        line += " &middot; accuracy A {:.1%}, B {:.1%} &middot; sign-test p={:.3f}".format(
            summary["accuracy_a"], summary["accuracy_b"], summary["p_value"])
    rows = []
    for r in results:
        rows.append(
            "<tr><td>{id}</td><td>{ma}</td><td>{fa:.0%}</td><td>{mb}</td><td>{fb:.0%}</td>"
            "<td>{ab}</td><td>{exp}</td>{ca}{cb}</tr>".format(
                id=escape(r["id"]), ma=escape(r["majority_a"]), fa=r["flip_a"],
                mb=escape(r["majority_b"]), fb=r["flip_b"],
                ab=("yes ({})".format(r["ab_layer"]) if r["ab_match"] else "NO"),
                exp=escape(r.get("expected", "")),
                ca=_cell_ok(r.get("correct_a")), cb=_cell_ok(r.get("correct_b"))))
    return _PAGE.format(verdict=summary["verdict"], model_a=escape(model_a), model_b=escape(model_b),
                        cases=summary["cases"], runs=summary["runs"], summary_line=line, rows="\n".join(rows))
