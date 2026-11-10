"""The paper's per-bin saving plot."""

from pathlib import Path


def saving_plot(rows, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    x = [(r["bin_index"] + 0.5) * 0.1 for r in rows]
    y = [100 * r["saving"] for r in rows]
    ax.plot(x, y, "o-")
    for a, b, r in zip(x, y, rows, strict=True):
        if r["thin"]:
            ax.axvspan(a - 0.05, a + 0.05, facecolor="none", edgecolor="0.75", hatch="///", linewidth=0)
        if r["ci_defined"]:
            ax.vlines(a, 100 * r["ci_lo"], 100 * r["ci_hi"])
        ax.annotate(str(r["n_tasks"]), (a, b), xytext=(0, 5), textcoords="offset points")
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.set(xlabel="Block-rounded mask ratio", ylabel="Step wall-time saving (%)", xlim=(0, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
