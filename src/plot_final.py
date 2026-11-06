"""Figure of the final test results (reporting only, no decisions)."""
import matplotlib.pyplot as plt
import pandas as pd

import config as C

if __name__ == "__main__":
    deg = pd.read_csv(C.OUTPUTS_DIR / "final_test" / "degradation_test.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for ax, (family, xlabel) in zip(axes, [("resolution", "effective resolution (px)"),
                                           ("blur", "Gaussian blur sigma (px)"), ("dark", "brightness factor")]):
        d = deg[deg.family == family]
        ax.plot(d.level, d.top1, "o-", label="top-1, all images")
        ax.plot(d.level, d.accepted_top1, "s--", label="top-1 on accepted (t=0.76)")
        ax.plot(d.level, d.coverage, "^:", label="coverage at t=0.76")
        ax.set(xlabel=xlabel, ylim=(0.4, 1), title=f"test: {family}")
        if family in ("resolution", "dark"):
            ax.invert_xaxis()
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.ROOT / "docs" / "figures" / "degradation_test.png", dpi=110)
