"""Plot historical reported scores without loading customer data."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

def main():
    with open(ROOT / "results/recorded_metrics.csv", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = list(range(len(rows)))
    for offset, metric, color in [(-0.18, "precision", "#2563eb"), (0.18, "recall", "#0d9488")]:
        bars = ax.bar([i + offset for i in x], [float(r[metric]) * 100 for r in rows], width=0.36, label=metric.title(), color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=9)
    ax.set_xticks(x, [r["model"].replace(" (", "\n(") for r in rows])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percent")
    ax.set_title("Loan acceptance: precision–recall tradeoff")
    ax.legend(loc="lower right")
    fig.text(0.5, 0.015, "Historical reported results; not a rerun of the cleaned code", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    (ROOT / "assets").mkdir(exist_ok=True)
    fig.savefig(ROOT / "assets/results.png", dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    main()
