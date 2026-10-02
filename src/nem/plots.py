"""Static figures for the README. Colors: dataviz reference palette (light mode), fixed slot order."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

MECH_ORDER = ("naive", "checked", "backfire", "repvote", "escrepvote")
MECH_LABEL = {"naive": "Naive", "checked": "Checked", "backfire": "Backfire",
              "repvote": "RepVote", "escrepvote": "EscRepVote"}
MECH_COLOR = {m: SERIES[i] for i, m in enumerate(MECH_ORDER)}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(ncols: int, width: float = 4.2, height: float = 3.8):
    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, height), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes[0]:
        _style(ax)
    return fig, list(axes[0])


def trajectory_panels(panels: dict[str, dict[str, list]], path: str, title: str) -> None:
    """panels: {panel title: {mechanism: [(fpr, tpr), ...] seed-averaged}}."""
    fig, axes = _fig(len(panels))
    for ax, (panel, by_mech) in zip(axes, panels.items()):
        ax.plot([0, 1], [0, 1], linestyle="--", color=INK_2, linewidth=1, label="Random")
        for mech in MECH_ORDER:
            traj = by_mech.get(mech)
            if not traj:
                continue
            pts = [(x, y) for x, y in traj if not (math.isnan(x) or math.isnan(y))]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=MECH_COLOR[mech], linewidth=2, label=MECH_LABEL[mech])
            ax.plot(xs[-1], ys[-1], marker="o", markersize=8, color=MECH_COLOR[mech],
                    markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False positive rate (good agents removed)", color=INK, fontsize=9)
        ax.set_title(panel, color=INK, fontsize=11)
    axes[0].set_ylabel("True positive rate (violators removed)", color=INK, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(title, color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def grouped_bars(groups: list[str], series: dict[str, list[tuple[float, float, float]]], path: str,
                 title: str, ylabel: str, ylim: float = 1.0) -> None:
    """series: {series label: [(value, ci_low, ci_high) per group]}. Values labeled on top of bars."""
    fig, axes = _fig(1, width=max(6.0, 1.3 * len(groups) * max(1, len(series)) / 2), height=4.2)
    ax = axes[0]
    n = len(series)
    width = 0.8 / n
    for i, (label, vals) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        ys = [v for v, _, _ in vals]
        err = [[max(0.0, v - lo) for v, lo, _ in vals], [max(0.0, hi - v) for v, _, hi in vals]]
        ax.bar(xs, ys, width=width * 0.92, color=SERIES[i], label=label, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
        for x, y in zip(xs, ys):
            if not math.isnan(y):
                ax.text(x, y + 0.02 * ylim, f"{y:.0%}" if ylim <= 1 else f"{y:.2f}",
                        ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, color=INK, fontsize=9)
    ax.set_ylim(0, ylim * 1.12)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))
