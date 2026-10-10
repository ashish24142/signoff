import argparse
import json
import sys
from pathlib import Path

import yaml

from . import core, report

def main(argv=None):
    p = argparse.ArgumentParser(prog="signoff",
        description="Statistical sign-off for the model swap.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="N-run A/B migration validation")
    r.add_argument("--suite", required=True, help="YAML suite: cases: [{id, prompt, expected?}]")
    r.add_argument("--model-a", required=True, help="current model, e.g. openai:gpt-4o")
    r.add_argument("--model-b", required=True, help="candidate model, e.g. anthropic:claude-sonnet-4-5")
    r.add_argument("--runs", type=int, default=5)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--judge", help="judge for the match layer: provider:model, or jev[:model]")
    r.add_argument("--judge-threshold", type=float, default=0.5,
                   help="jev judge only: minimum same-information probability to count a match")
    r.add_argument("--out", default="signoff-report.html")

    st = sub.add_parser("stability", help="N-run flakiness measurement for a single model")
    st.add_argument("--suite", required=True,
                    help="YAML suite: cases with 'prompt' or 'sections: [{name, text}]'")
    st.add_argument("--model", required=True, help="provider:model, e.g. openai:gpt-4o")
    st.add_argument("--runs", type=int, default=10)
    st.add_argument("--temperature", type=float, default=0.0)
    st.add_argument("--ablate", action="store_true",
                    help="leave-one-out section ablation to attribute flips (needs 'sections')")
    st.add_argument("--gate", type=float,
                    help="max acceptable per-case flip rate; exit 1 if exceeded (CI gate)")
    st.add_argument("--out", default="stability-report.html")
    args = p.parse_args(argv)

    suite = yaml.safe_load(Path(args.suite).read_text(encoding="utf-8"))
    out = Path(args.out)

    if args.cmd == "stability":
        results = core.stability_suite(suite["cases"], args.model, runs=args.runs,
                                       temperature=args.temperature, ablate=args.ablate)
        summary = core.stability_summary(results, args.runs, gate=args.gate)
        out.write_text(report.build_stability_html(summary, results, args.model), encoding="utf-8")
        out.with_suffix(".json").write_text(
            json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
        print(f"cases={summary['cases']} runs={summary['runs']} "
              f"mean_flip={summary['mean_flip']:.1%} worst_flip={summary['worst_flip']:.1%} "
              f"unstable_cases={summary['unstable_cases']}")
        print(f"verdict: {summary['verdict']}   report: {out} / {out.with_suffix('.json')}")
        return 1 if summary["verdict"] == "FAIL" else 0

    if args.judge and (args.judge == "jev" or args.judge.startswith("jev:")):
        jev_model = args.judge.partition(":")[2] or "jev-1.13.0"
        judge = core.jev_judge(jev_model, threshold=args.judge_threshold)
    elif args.judge:
        judge = core.llm_judge(args.judge)
    else:
        judge = None
    results = core.run_suite(suite["cases"], args.model_a, args.model_b,
                             runs=args.runs, temperature=args.temperature, judge=judge)
    summary = core.summarize(results, args.runs)

    out.write_text(report.build_html(summary, results, args.model_a, args.model_b), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")

    print(f"cases={summary['cases']} agreement={summary['agreement']:.1%} "
          f"flip_a={summary['flip_a']:.1%} flip_b={summary['flip_b']:.1%}")
    if "accuracy_a" in summary:
        print(f"accuracy: A={summary['accuracy_a']:.1%} B={summary['accuracy_b']:.1%} "
              f"p={summary['p_value']:.3f}")
    print(f"verdict: {summary['verdict']}   report: {out} / {out.with_suffix('.json')}")
    return 1 if summary["verdict"] == "FAIL" else 0

if __name__ == "__main__":
    sys.exit(main())
