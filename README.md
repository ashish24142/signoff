# signoff

**Statistical sign-off for the model swap.** N-run, nondeterminism-aware A/B equivalence testing for LLM migrations, with an audit-ready report.

Model deprecations force migrations on someone else's schedule. "We eyeballed ten outputs" is not evidence a regulated enterprise can take to a risk committee. `signoff` runs your real prompt suite on the old and new model N times each, measures per-case flip rates (nondeterminism), scores answers through layered matching (exact, normalized, numeric, then LLM-judge), and applies a paired sign test before declaring the swap safe.

Vendor-neutral by design: it judges an OpenAI to Anthropic migration with exactly the same rigor as the reverse.

## Install

```bash
pip install -e .
export OPENAI_API_KEY=...      # for openai: models and/or the judge
export ANTHROPIC_API_KEY=...   # for anthropic: models
```

## Use

```bash
signoff run \
  --suite examples/extraction.yaml \
  --model-a openai:gpt-4o \
  --model-b anthropic:claude-sonnet-4-5 \
  --runs 5 \
  --judge openai:gpt-4o-mini \
  --out report.html
```

Output: a one-line verdict in the terminal, plus `report.html` (human) and `report.json` (machine).

```
cases=48 agreement=95.8% flip_a=2.1% flip_b=1.7%
accuracy: A=93.8% B=95.8% p=0.512
verdict: PASS   report: report.html / report.json
```

**Verdict logic:** with ground truth (`expected` per case), FAIL only when model B is less accurate *and* the regression is statistically significant (exact sign test on discordant pairs, p <= 0.05). Without ground truth, 90%+ majority-answer agreement passes; below that, REVIEW.

## Suite format

```yaml
cases:
  - id: facility-amount
    prompt: "Extract the total facility amount. ..."
    expected: "$250,000,000"   # optional; omit for agreement-only mode
```

## Why N runs?

Single-run comparisons confuse nondeterminism with regression. A prompt that flips answers 30% of the time will "regress" or "improve" by luck. `signoff` reports the flip rate per case per model, so instability is visible before it's misattributed to the migration.

## Roadmap

- Semantic match layer (embeddings) between numeric and judge
- Governance evidence annex: mapping results to ISO/IEC 42001 change control, NIST AI RMF MEASURE, EU AI Act Annex IV
- Adapters: Bedrock, Azure OpenAI, Gemini
- Equivalence bands (TOST) as an alternative to the sign test

## License

Apache-2.0
