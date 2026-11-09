"""Static charts from results.json, for the README and the dashboard.

Colours: categorical slots 1-3 of the validated reference palette (blue, orange, aqua),
checked with the dataviz validator (CVD and normal-vision separation pass; aqua is below
3:1 contrast on the surface, so every aqua mark carries a visible value label and the
dashboard ships a table view). Text uses ink colours, never series colours.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SHORT = {"openai/gpt-oss-120b": "gpt-oss-120b", "openai/gpt-oss-20b": "gpt-oss-20b",
         "gemini-3.6-flash": "gemini-3.6-flash"}


def _style(ax, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold",
                 pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2)


def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path, legend_ax=None, ncol=3):
    """Lay out and save. With legend_ax, its legend goes in a reserved band under the plot."""
    if legend_ax is not None:
        h, l = legend_ax.get_legend_handles_labels()
        fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=ncol, frameon=False,
                   fontsize=8, labelcolor=INK2, handletextpad=0.3, columnspacing=1.5)
        band = 0.55 / fig.get_size_inches()[1]
        fig.tight_layout(rect=(0, band, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _provisional(res):
    return any(not m["progress"]["complete"] for m in res["models"].values())


def accuracy_vs_baselines(res: dict, path: Path):
    """Each model's accuracy (95% CI) next to TF-IDF and majority on the same reports."""
    rows = []
    for name, m in res["models"].items():
        for cond, e in m.get("main", {}).items():
            a = e["all"]
            rows.append((f"{SHORT.get(name, name)}\n{cond.replace('_', ' ')}", a))
    if not rows:
        return False
    fig, ax = _fig(8, 0.75 * len(rows) + 1.6)
    y = np.arange(len(rows))[::-1]
    for yi, (_, a) in zip(y, rows):
        lo, hi = a["accuracy_ci95"]
        ax.plot([lo, hi], [yi, yi], color=BLUE, linewidth=2, solid_capstyle="round")
        ax.scatter(a["accuracy"], yi, s=60, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.text(hi + 0.012, yi, f"{a['accuracy']:.2f}", va="center", fontsize=8, color=INK)
        ax.scatter(a["tfidf_accuracy_same_reports"], yi - 0.22, marker="D", s=34, color=AQUA,
                   zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.text(a["tfidf_accuracy_same_reports"] + 0.012, yi - 0.22,
                f"{a['tfidf_accuracy_same_reports']:.2f}", va="center", fontsize=7, color=INK2)
        ax.scatter(a["majority_accuracy_same_reports"], yi - 0.22, marker="|", s=120, color=GREY,
                   zorder=3, linewidth=2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("accuracy against the ASRS analyst's Primary Problem code", color=INK2, fontsize=9)
    ax.scatter([], [], color=BLUE, s=40, label="LLM (line = 95% CI)")
    ax.scatter([], [], color=AQUA, marker="D", s=30, label="TF-IDF + logistic regression, same reports")
    ax.scatter([], [], color=GREY, marker="|", s=100, label="always 'most common class', same reports")
    _style(ax, "Accuracy vs baselines" + ("  (PROVISIONAL: still collecting)" if _provisional(res) else ""))
    _save(fig, path, legend_ax=ax, ncol=3)
    return True


def memorisation(res: dict, path: Path):
    """Before-minus-after-cutoff accuracy gap, raw and net of the TF-IDF difficulty control."""
    rows = []
    for name, m in res["models"].items():
        for cond, e in m.get("main", {}).items():
            mem = e.get("memorisation", {})
            if "llm_gap" in mem:
                rows.append((f"{SHORT.get(name, name)}\n{cond.replace('_', ' ')}\n"
                             f"(n={mem['n_pre']}+{mem['n_post']})", mem))
    if not rows:
        return False
    fig, ax = _fig(8, 0.85 * len(rows) + 1.6)
    y = np.arange(len(rows))[::-1]
    for yi, (_, mem) in zip(y, rows):
        for off, key, col in [(0.14, "llm_gap", ORANGE), (-0.14, "adjusted_gap", BLUE)]:
            g = mem[key]
            ax.plot(g["ci95"], [yi + off] * 2, color=col, linewidth=2, solid_capstyle="round")
            ax.scatter(g["estimate"], yi + off, s=50, color=col, zorder=3, edgecolor=SURFACE,
                       linewidth=2)
            ax.text(g["ci95"][1] + 0.02, yi + off, f"{g['estimate']:+.2f}", va="center",
                    fontsize=8, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8.5, color=INK)
    ax.set_xlabel("accuracy before cutoff − accuracy after (positive = better on reports it may have seen)",
                  color=INK2, fontsize=9)
    ax.scatter([], [], color=ORANGE, s=40, label="raw gap")
    ax.scatter([], [], color=BLUE, s=40, label="gap net of TF-IDF's gap on the same reports")
    _style(ax, "Memorisation: does accuracy fall after the training cutoff?"
           + ("  (PROVISIONAL)" if _provisional(res) else ""), "Lines are 95% bootstrap CIs")
    _save(fig, path, legend_ax=ax, ncol=2)
    return True


def reliability(res: dict, path: Path):
    """Reliability diagrams (no-retrieval condition), one panel per model."""
    panels = [(n, m["main"]["no_retrieval"]["all"]) for n, m in res["models"].items()
              if "no_retrieval" in m.get("main", {}) and "reliability_bins" in m["main"]["no_retrieval"]["all"]]
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.8), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, a) in zip(axes[0], panels):
        ax.plot([0, 1], [0, 1], color=GREY, linewidth=1, linestyle="--")
        bins = [b for b in a["reliability_bins"] if b["count"]]
        xs = [b["mean_conf"] for b in bins]
        ys = [b["accuracy"] for b in bins]
        sizes = [20 + 180 * b["count"] / a["n"] for b in bins]
        ax.plot(xs, ys, color=BLUE, linewidth=2)
        ax.scatter(xs, ys, s=sizes, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        _style(ax, SHORT.get(name, name))
        ax.text(0.03, 0.92, f"ECE {a['ece']:.2f}\nn={a['n']}", transform=ax.transAxes,
                fontsize=8, color=INK, va="top")
        ax.set_xlabel("stated confidence", color=INK2, fontsize=8)
    axes[0][0].set_ylabel("fraction correct", color=INK2, fontsize=8)
    fig.suptitle("Calibration: when it says 80%, is it right 80% of the time?"
                 + ("  (PROVISIONAL)" if _provisional(res) else ""),
                 x=0.02, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.text(0.02, 0.012, "Dashed line = perfect calibration. Below it = overconfident. "
             "Dot size = reports in that confidence bin.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


def probes(res: dict, path: Path):
    """Probe answer-change rate vs the identical-prompt noise floor."""
    panels = [(n, m) for n, m in res["models"].items() if m.get("probes")]
    panels = [(n, m) for n, m in panels if any("answer_change_rate" in p for p in m["probes"].values())]
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.0), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, m) in zip(axes[0], panels):
        nf = m.get("noise_floor", {}).get("pairwise_disagreement")
        if nf:
            ax.axvspan(nf["ci95"][0], nf["ci95"][1], color=GRID, zorder=0)
            ax.axvline(nf["estimate"], color=INK2, linewidth=1)
            ax.text(nf["estimate"], 1.02, " noise floor", transform=ax.get_xaxis_transform(),
                    fontsize=7.5, color=INK2)
        items = [(p, v) for p, v in m["probes"].items() if "answer_change_rate" in v]
        for yi, (p, v) in enumerate(items[::-1]):
            c = v["answer_change_rate"]
            ax.plot(c["ci95"], [yi, yi], color=BLUE, linewidth=2, solid_capstyle="round")
            ax.scatter(c["estimate"], yi, s=50, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=2)
            ax.text(c["ci95"][1] + 0.01, yi, f"{c['estimate']:.2f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(range(len(items)), [f"{p} (n={v['n']})" for p, v in items[::-1]], color=INK)
        ax.set_xlim(0, max(0.5, max(v["answer_change_rate"]["ci95"][1] for _, v in items) + 0.08))
        _style(ax, SHORT.get(name, name))
        ax.set_xlabel("share of answers that changed vs the unprobed prompt", color=INK2, fontsize=8)
    fig.suptitle("Probes vs noise floor" + ("  (PROVISIONAL)" if _provisional(res) else ""),
                 x=0.02, ha="left", fontsize=11, color=INK, fontweight="bold")
    _save(fig, path)
    return True


def progress(done: dict, planned: dict, path: Path):
    """Share of each model's planned jobs finished: one bar per model."""
    names = [n for n in planned if planned[n]]
    if not names:
        return False
    fig, ax = _fig(8, 0.55 * len(names) + 1.0)
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        frac = done.get(n, 0) / planned[n]
        ax.barh(yi, 1, height=0.5, color=GRID)
        ax.barh(yi, frac, height=0.5, color=BLUE)
        ax.text(1.01, yi, f"{done.get(n, 0)} / {planned[n]}  ({frac:.0%})", va="center",
                fontsize=8, color=INK)
    ax.set_yticks(y, [SHORT.get(n, n) for n in names], color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(ax, "Experiment progress: planned jobs finished")
    ax.grid(axis="x", visible=False)
    _save(fig, path)
    return True


def all_charts(res: dict, traces, plan_counts: dict, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for fn, fname in [(accuracy_vs_baselines, "accuracy.png"), (memorisation, "memorisation.png"),
                      (reliability, "calibration.png"), (probes, "probes.png")]:
        if fn(res, out_dir / fname):
            made.append(fname)
    done = {n: m["progress"]["done"] for n, m in res["models"].items()}
    if progress(done, plan_counts, out_dir / "progress.png"):
        made.append("progress.png")
    return made
