"""All figures, one function each.

Every function takes plain tables (pandas DataFrames) and an output path, so
the same code works whether the numbers come from run folders or from the
committed CSVs in results/.

Colours: model sizes are ordered, so they use one blue ramp from light (small)
to dark (large). Domains are categories and use three fixed hues. Both sets
were checked for colour-blind separation.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

SIZE_COLORS = {"S1": "#86b6ef", "S2": "#5598e7", "S3": "#2a78d6", "S4": "#1c5cab", "S5": "#104281"}
DOMAIN_COLORS = {"image": "#2a78d6", "audio": "#eb6834", "text": "#1baf7a", "mean": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def _tokens_label(x, _pos=None) -> str:
    """1500000 -> "1.5M", so log axes stay readable."""
    for unit, name in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if x >= unit:
            return f"{x / unit:g}{name}"
    return f"{x:g}"


def _log_tokens_axis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))  # labels even on axes shorter than a decade
    ax.xaxis.set_major_formatter(FuncFormatter(_tokens_label))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("tokens seen")


def load_run_curves(run_dirs: list[Path]):
    """Read metrics.csv and summary.json of several runs into two tidy tables.

    train: one row per logged step (size, tokens, train_loss).
    held_out: one row per evaluation point and domain (size, tokens, domain, loss),
    using the test split.
    """
    train, held_out = [], []
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        size = summary["config"]["size"]
        metrics = pd.read_csv(run_dir / "metrics.csv")
        train.append(metrics.assign(size=size)[["size", "tokens", "train_loss"]])
        for point in summary["curve"]:
            for domain, loss in point["test"].items():
                held_out.append({"size": size, "tokens": point["tokens"], "domain": domain, "loss": loss})
    return pd.concat(train), pd.DataFrame(held_out)


def learning_curves(train: pd.DataFrame, held_out: pd.DataFrame, title: str, out: Path, floors: dict | None = None):
    """Figure 1: training loss and held-out loss per domain against tokens seen.

    `floors` optionally maps "image"/"audio" to the caption loss of a model
    that cannot see the input, drawn as a dashed line for reference.
    """
    domains = ["image", "audio", "text"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
    for size, df in train.groupby("size"):
        # The raw batch loss is noisy because each batch mixes domains differently.
        smooth = df["train_loss"].rolling(5, min_periods=1).mean()
        axes[0].plot(df["tokens"], smooth, color=SIZE_COLORS[size], label=size)
    axes[0].set_title("training loss (all domains)")
    axes[0].set_ylabel("loss (nats per token)")
    axes[0].legend(title="model")

    for ax, domain in zip(axes[1:], domains):
        for size, df in held_out[held_out["domain"] == domain].groupby("size"):
            ax.plot(df["tokens"], df["loss"], marker="o", markersize=4, color=SIZE_COLORS[size])
        if floors and domain in floors:
            ax.axhline(floors[domain], color=MUTED, linestyle="--", linewidth=1)
            ax.annotate("no-input floor", (0.02, floors[domain]), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", color=MUTED, fontsize=8)
        ax.set_title(f"{domain}: test loss")
    for ax in axes:
        _log_tokens_axis(ax)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
