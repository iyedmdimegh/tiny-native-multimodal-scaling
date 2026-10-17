"""All figures, one function each.

Every function takes plain tables (pandas DataFrames, read from the committed
CSVs in results/) and an output path. scripts/make_figures.py calls them.

Colours: model sizes and token budgets are both ordered, so each gets its own
one-hue ramp from light (small) to dark (large): blue for sizes, orange for
budgets. Domains are categories and use fixed hues. The ramps and the domain
hues were checked with a colour-blind palette validator.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from tnms.fit import scaling

SIZE_COLORS = {"S1": "#86b6ef", "S2": "#5598e7", "S3": "#2a78d6", "S4": "#1c5cab", "S5": "#104281"}
BUDGET_RAMP = ["#f09a66", "#e0743c", "#b9541f", "#8c3c13", "#5f2709"]
DOMAIN_COLORS = {"image": "#2a78d6", "audio": "#eb6834", "text": "#1baf7a", "mean": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TARGETS = ["image", "audio", "text", "mean"]

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


def _short(x, _pos=None) -> str:
    """1500000 -> "1.5M", so log axes stay readable."""
    for unit, name in ((1e18, "E"), (1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if x >= unit:
            return f"{x / unit:g}{name}"
    return f"{x:g}"


def _log_axis(ax, label: str, axis: str = "x", subs=(1, 2, 5)):
    target = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    target.set_major_locator(LogLocator(subs=subs))  # 2 and 5 give labels even on axes shorter than a decade
    target.set_major_formatter(FuncFormatter(_short))
    target.set_minor_formatter(NullFormatter())
    (ax.set_xlabel if axis == "x" else ax.set_ylabel)(label)


def budget_colors(budgets) -> dict:
    return dict(zip(sorted(budgets), BUDGET_RAMP))


def _floor_line(ax, floors: dict, domain: str):
    if domain in floors:
        ax.axhline(floors[domain], color=MUTED, linestyle="--", linewidth=1)


def _save(fig, title: str, out: Path):
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def learning_curves(curves: pd.DataFrame, size: str, floors: dict, out: Path):
    """Figure 1: test loss during training for one model size, one line per budget D.

    Each budget is its own run with its own schedule, so the lines are not
    pieces of one long run. Points are the evaluations at 25/50/75/100%.
    """
    df = curves[(curves["size"] == size) & (curves["seed"] == 0) & (curves["split"] == "test")]
    colors = budget_colors(df["D_budget"].unique())
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
    for ax, domain in zip(axes, TARGETS):
        for budget, run in df[df["domain"] == domain].groupby("D_budget"):
            ax.plot(run["tokens"], run["loss"], marker="o", markersize=4, color=colors[budget], label=_short(budget))
        _floor_line(ax, floors, domain)
        ax.set_title(f"{domain}: test loss")
        _log_axis(ax, "tokens seen")
    axes[0].set_ylabel("loss (nats per text token)")
    axes[-1].legend(title="budget D", fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    _save(fig, f"Test loss during training, model {size}, one run per token budget. Dashed: no-input floor", out)


def _noise_errors(noise: pd.DataFrame, target: str) -> dict:
    """{(size, D_budget): std} for the cells that were run with several seeds."""
    if noise is None or noise.empty:
        return {}
    return {(r["size"], r["D_budget"]): r[f"test_{target}_std"] for _, r in noise.iterrows()}


def loss_vs_data(grid: pd.DataFrame, fits: dict, noise: pd.DataFrame, floors: dict, out: Path):
    """Figure 2: test loss against D for each size, with the fitted law as lines."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    d_line = np.geomspace(grid["D"].min() * 0.9, grid["D"].max() * 1.1, 100)
    for ax, target in zip(axes, TARGETS):
        errors = _noise_errors(noise, target)
        params = np.array(fits[target]["params"])
        for size, rows in grid.groupby("size"):
            n = rows["N"].iloc[0]
            yerr = [errors.get((size, b), 0.0) for b in rows["D_budget"]]
            ax.errorbar(rows["D"], rows[target], yerr=yerr, fmt="o", markersize=4, color=SIZE_COLORS[size],
                        capsize=2, label=size)
            ax.plot(d_line, scaling.predict_nd(params, np.full_like(d_line, n), d_line), color=SIZE_COLORS[size],
                    linewidth=1.2)
        _floor_line(ax, floors, target)
        ax.set_title(f"{target}: test loss")
        _log_axis(ax, "training tokens D")
    axes[0].set_ylabel("loss (nats per text token)")
    axes[-1].legend(title="model", fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    _save(fig, "Test loss against training tokens. Points are runs, lines the fitted L(N, D), dashed the no-input floor", out)


def loss_vs_params(grid: pd.DataFrame, fits: dict, floors: dict, out: Path):
    """Figure 3: test loss against N for each budget, with the fitted law as lines."""
    colors = budget_colors(grid["D_budget"].unique())
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    n_line = np.geomspace(grid["N"].min() * 0.8, grid["N"].max() * 1.2, 100)
    for ax, target in zip(axes, TARGETS):
        params = np.array(fits[target]["params"])
        for budget, rows in grid.groupby("D_budget"):
            ax.plot(rows["N"], rows[target], "o", markersize=4, color=colors[budget], label=_short(budget))
            d = rows["D"].mean()
            ax.plot(n_line, scaling.predict_nd(params, n_line, np.full_like(n_line, d)), color=colors[budget],
                    linewidth=1.2)
        _floor_line(ax, floors, target)
        ax.set_title(f"{target}: test loss")
        _log_axis(ax, "non-embedding parameters N", subs=(1,))  # N spans 2+ decades
    axes[0].set_ylabel("loss (nats per text token)")
    axes[-1].legend(title="budget D", fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    _save(fig, "Test loss against model size. Points are runs, lines the fitted L(N, D), dashed the no-input floor", out)


def holdout_predictions(holdout: pd.DataFrame, fits: dict, out: Path):
    """Figure 4: predicted against observed loss for the runs left out of the fit."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, (name, rows) in zip(axes, holdout.groupby("holdout", sort=False)):
        for target, t_rows in rows.groupby("target", sort=False):
            err = fits[target]["holdout_mare"][name]
            ax.plot(t_rows["observed"], t_rows["law"], "o", markersize=5, color=DOMAIN_COLORS[target],
                    label=f"{target}: law {err['law']:.1%}, baseline {err['baseline']:.1%}")
            ax.plot(t_rows["observed"], t_rows["baseline"], "x", markersize=5, color=DOMAIN_COLORS[target])
        lo = min(rows[["observed", "law", "baseline"]].min()) * 0.9
        hi = max(rows[["observed", "law", "baseline"]].max()) * 1.1
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--")
        _log_axis(ax, "observed test loss")
        _log_axis(ax, "predicted test loss", axis="y")
        ax.set_title(f"held out: runs with the {name}")
        ax.legend(fontsize=7, title="o law, x baseline (mean abs. rel. error)", title_fontsize=7, loc="upper left")
    _save(fig, "Predicted against observed test loss for runs left out of the fit", out)


def compute_frontier(grid: pd.DataFrame, fits: dict, out: Path, target: str = "mean"):
    """Figure 5: loss against compute C = 6ND, the empirical frontier and the fitted optimum.

    The frontier joins the runs that beat every run with less compute. The
    dashed line is the fitted law evaluated at the compute-optimal N*(C), D*(C).
    """
    df = grid.assign(C=6 * grid["N"] * grid["D"]).sort_values("C")
    frontier = df[df[target] < df[target].cummin().shift(fill_value=np.inf)]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for size, rows in df.groupby("size"):
        ax.plot(rows["C"], rows[target], "o", markersize=5, color=SIZE_COLORS[size], label=size)
    ax.plot(frontier["C"], frontier[target], color=INK, linewidth=1.2, label="empirical frontier")
    params = np.array(fits[target]["params"])
    c_line = np.geomspace(df["C"].min(), df["C"].max(), 100)
    n_opt, d_opt = scaling.compute_optimal(params, c_line)
    ax.plot(c_line, scaling.predict_nd(params, n_opt, d_opt), color=MUTED, linestyle="--", linewidth=1.2,
            label="fitted optimum")
    _log_axis(ax, "training compute C = 6ND (FLOPs)")
    ax.set_ylabel(f"{target} test loss (nats per text token)")
    ax.legend(fontsize=8)
    _save(fig, f"{target.capitalize()} test loss against training compute", out)


def _ternary(H: np.ndarray):
    """Shares (image, audio, text) to x, y in a triangle: image bottom left, audio bottom right, text on top."""
    return H[:, 1] + H[:, 2] / 2, H[:, 2] * np.sqrt(3) / 2


def mixture_sweep(sweep: pd.DataFrame, fit: dict, out: Path):
    """Figure 6: test loss of each domain against its own share, and the predicted mean loss over the simplex.

    The heatmap uses the law fitted on validation loss, the same one used to
    choose the mixtures, so it shows what the choice was based on.
    """
    from tnms.fit import mixture

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    for ax, (i, domain) in zip(axes, enumerate(mixture.DOMAINS)):
        ax.plot(sweep[f"h_{domain}"], sweep[f"test_{domain}"], "o", markersize=5, color=DOMAIN_COLORS[domain])
        ax.set_xlabel(f"{domain} share of training tokens")
        ax.set_title(f"{domain}: test loss")
    axes[0].set_ylabel("loss (nats per text token)")

    grid = mixture.simplex_grid(step=0.01)
    pred = np.mean([mixture.predict(np.array(fit["params"][d]), grid, i) for i, d in enumerate(mixture.DOMAINS)],
                   axis=0)
    ax = axes[3]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", list(SIZE_COLORS.values()))
    x, y = _ternary(grid)
    # The law blows up near the corners, so the colour scale stops at the 90th percentile
    # (marked by the arrow on the colour bar) to keep the middle of the triangle readable.
    points = ax.scatter(x, y, c=pred, cmap=cmap, s=4, marker="h", linewidths=0, vmax=np.quantile(pred, 0.9))
    fig.colorbar(points, ax=ax, label="predicted mean val loss", shrink=0.85, extend="max")
    tx, ty = _ternary(sweep[[f"h_{d}" for d in mixture.DOMAINS]].to_numpy())
    ax.plot(tx, ty, "o", markersize=5, markerfacecolor="white", markeredgecolor=INK, label="trained mixtures")
    for target, marker in (("mean", "*"), ("image", "D")):
        bx, by = _ternary(np.array([fit["optimum"][target]["h"]]))
        ax.plot(bx, by, marker, markersize=9 if marker == "*" else 6, color=INK, label=f"predicted best {target}")
    for label, (vx, vy, ha) in {"image": (0, -0.07, "center"), "audio": (1, -0.07, "center"),
                                "text": (0.5, np.sqrt(3) / 2 + 0.03, "center")}.items():
        ax.annotate(label, (vx, vy), ha=ha, color=MUTED, fontsize=8, annotation_clip=False)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, columnspacing=0.8)
    ax.set_title("predicted mean val loss over mixtures", pad=24)
    _save(fig, "Mixture sweep at fixed model size and token budget", out)


def mixture_validation(validation: pd.DataFrame, noise_std: dict, out: Path):
    """Figure 7: test loss of the fresh runs at uniform and at the two predicted-best mixtures.

    Error bars are one standard deviation across seeds of the uniform mixture
    at the same size and budget, so they show how big a difference has to be
    before it means anything. Dots, not bars: the differences are small and a
    bar would have to start at zero.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    colors = {"uniform": MUTED, "best mean": DOMAIN_COLORS["image"], "best image": DOMAIN_COLORS["image"]}
    for ax, target in zip(axes, ("mean", "image")):
        y = range(len(validation))
        for k, (_, r) in enumerate(validation.iterrows()):
            ax.errorbar(r[f"test_{target}"], k, xerr=noise_std[target], fmt="o", markersize=7, capsize=3,
                        color=colors[r["label"]])
        ax.set_yticks(list(y), [f"{r['label']}\n{tuple(round(r[f'h_{d}'], 2) for d in ('image', 'audio', 'text'))}"
                                for _, r in validation.iterrows()], fontsize=8)
        ax.set_xlabel(f"{target} test loss (nats per text token)")
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    _save(fig, "Fresh runs at the predicted-best mixtures and at uniform, with seed spread", out)
