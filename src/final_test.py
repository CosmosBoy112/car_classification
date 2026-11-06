"""Step 9: the one and only test evaluation.

Reads the frozen choice from outputs/final_choice.json (written by notebooks/03_analysis.ipynb on val),
then evaluates that model on the official test split, clean and degraded, with the chosen threshold.
Nothing here feeds back into any decision.

    python src/final_test.py
"""
import json

import pandas as pd

import config as C
from evaluate import evaluate_degradations

if __name__ == "__main__":
    choice = json.loads((C.OUTPUTS_DIR / "final_choice.json").read_text())
    run, t = choice["final_run"], choice["threshold"]
    print(f"final model {run}, threshold {t}")

    deg = evaluate_degradations(run, "test", allow_test=True, threshold=t)
    m = json.loads((C.OUTPUTS_DIR / run / "metrics_test.json").read_text())
    clean = deg[deg.condition == "res224"].iloc[0]
    summary = {"run": run, "threshold": t, **{f"test_{k}": v for k, v in m.items()},
               "test_coverage_at_threshold": clean.coverage, "test_accepted_top1_at_threshold": clean.accepted_top1}

    out = C.OUTPUTS_DIR / "final_test"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    deg.to_csv(out / "degradation_test.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(deg[["family", "condition", "top1", "top5", "coverage", "accepted_top1"]].round(4).to_string(index=False))
