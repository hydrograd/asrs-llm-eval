"""Static figures from results.json, for the README and the dashboard.

Academic figure style: serif type (Charis SIL, the SIL release of Bitstream Charter, shipped
in app/static/fonts under the OFL), no titles inside the image (each figure is captioned where
it is shown), paired panels (a)/(b) on a shared axis, and a "provisional" stamp while the
experiment is still collecting.

Palette, aviation-derived and checked with the dataviz validator on the paper surface
(all-pairs CVD and normal-vision separation pass; every colour >= 3:1 contrast):
  aviation blue #2f6fa3 (the LLM), International Orange #e05a1e (raw / warning-type
  quantity), runway-light green #3a9a6a (TF-IDF baseline). Text is ink, never series colour.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

PAPER, INK, INK2, RULE, GRID = "#f8f6f0", "#1a2433", "#4f5866", "#9aa0a8", "#e3dfd5"
BLUE, ORANGE, GREEN, GREY = "#2f6fa3", "#e05a1e", "#3a9a6a", "#8d8f93"
SHORT = {"openai/gpt-oss-120b": "gpt-oss-120b", "openai/gpt-oss-20b": "gpt-oss-20b",
         "gemini-3.6-flash": "gemini-3.6-flash"}
FONT_DIR = Path(__file__).resolve().parents[2] / "app/static/fonts"


def _setup_fonts():
    family = "serif"
    if FONT_DIR.exists():
        for f in FONT_DIR.glob("*.ttf"):
            font_manager.fontManager.addfont(str(f))
        family = "Charis SIL"
    plt.rcParams.update({
        "font.family": family, "font.serif": ["Charis SIL", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 9,
        "axes.edgecolor": RULE, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK, "xtick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 0, "figure.facecolor": PAPER,
        "axes.facecolor": PAPER, "savefig.facecolor": PAPER, "legend.frameon": False,
    })


_setup_fonts()


def _axes(ax, xgrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(False)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _panel(ax, letter, label):
    ax.text(0, 1.03, f"({letter})", transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=INK, va="bottom")
    ax.annotate(label, xy=(0, 1.03), xycoords="axes fraction", xytext=(21, 0),
                textcoords="offset points", fontsize=9.5, fontstyle="italic", color=INK, va="bottom")


def _stamp(fig, res, n_note=""):
    if any(not m["progress"]["complete"] for m in res["models"].values()):
        fig.text(0.995, 0.995, f"provisional{n_note}: collection in progress", ha="right",
                 va="top", fontsize=7.5, fontstyle="italic", color=ORANGE)


def _legend(fig, handles, ncol):
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=ncol,
               fontsize=8, labelcolor=INK2, handletextpad=0.4, columnspacing=1.6)


def _save(fig, path, bottom=0.0, top=0.95):
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _h(marker, color, label, **kw):
    return plt.Line2D([], [], marker=marker, color=color, linestyle="none", label=label, **kw)


def _conditions(res):
    conds = []
    for m in res["models"].values():
        for c in m.get("main", {}):
            if c not in conds:
                conds.append(c)
    return sorted(conds, key=lambda c: c != "no_retrieval")


# ------------------------------------------------------------------- figure: accuracy

def accuracy_vs_baselines(res: dict, path: Path):
    conds = _conditions(res)
    if not conds:
        return False
    models = list(res["models"])
    fig, axes = plt.subplots(1, len(conds), figsize=(2.9 * len(conds) + 1.3, 0.55 * len(models) + 1.5),
                             sharey=True, squeeze=False)
    y = np.arange(len(models))[::-1]
    for i, (ax, cond) in enumerate(zip(axes[0], conds)):
        for yi, name in zip(y, models):
            e = res["models"][name].get("main", {}).get(cond)
            if not e:
                continue
            a = e["all"]
            lo, hi = a["accuracy_ci95"]
            ax.plot([lo, hi], [yi, yi], color=BLUE, linewidth=1.6, solid_capstyle="butt")
            ax.plot(a["accuracy"], yi, "o", color=BLUE, markersize=6.5, markeredgecolor=PAPER,
                    markeredgewidth=1.2, zorder=3)
            ax.plot(a["tfidf_accuracy_same_reports"], yi, "D", color=GREEN, markersize=5,
                    markeredgecolor=PAPER, markeredgewidth=1, zorder=3)
            ax.plot(a["majority_accuracy_same_reports"], yi, "|", color=GREY, markersize=11,
                    markeredgewidth=1.6, zorder=2)
            ax.text(1.0, yi, f"{a['accuracy']:.2f}", va="center", ha="left", fontsize=8,
                    color=INK, transform=ax.get_yaxis_transform())
        _axes(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.set_yticks(y, [SHORT.get(m, m) for m in models])
        ax.set_xlabel("agreement with ASRS analyst (accuracy)")
        _panel(ax, "abcd"[i], cond.replace("_", " "))
    _legend(fig, [_h("o", BLUE, "LLM, with 95% CI", markersize=6),
                  _h("D", GREEN, "TF-IDF + logistic regression, same reports", markersize=5),
                  _h("|", GREY, "most common class, same reports", markersize=10, markeredgewidth=1.6)], 3)
    _stamp(fig, res)
    _save(fig, path, bottom=0.12)
    return True


# --------------------------------------------------------------- figure: memorisation

def memorisation(res: dict, path: Path):
    conds = _conditions(res)
    models = [n for n, m in res["models"].items()
              if any("llm_gap" in e.get("memorisation", {}) for e in m.get("main", {}).values())]
    if not models:
        return False
    fig, axes = plt.subplots(1, len(conds), figsize=(2.9 * len(conds) + 1.4, 0.75 * len(models) + 1.5),
                             sharey=True, sharex=True, squeeze=False)
    y = np.arange(len(models))[::-1]
    for i, (ax, cond) in enumerate(zip(axes[0], conds)):
        ax.axvline(0, color=INK2, linewidth=0.8)
        for yi, name in zip(y, models):
            mem = res["models"][name].get("main", {}).get(cond, {}).get("memorisation", {})
            if "llm_gap" not in mem:
                continue
            for off, key, col, mk in [(0.15, "llm_gap", ORANGE, "o"), (-0.15, "adjusted_gap", BLUE, "s")]:
                g = mem[key]
                ax.plot(g["ci95"], [yi + off] * 2, color=col, linewidth=1.6, solid_capstyle="butt")
                ax.plot(g["estimate"], yi + off, mk, color=col, markersize=6, markeredgecolor=PAPER,
                        markeredgewidth=1.2, zorder=3)
            ax.text(1.0, yi, f"n={mem['n_pre']}+{mem['n_post']}", va="center", ha="left",
                    fontsize=7.5, color=INK2, transform=ax.get_yaxis_transform())
        _axes(ax)
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.set_yticks(y, [SHORT.get(m, m) for m in models])
        ax.set_xlabel("accuracy before − after cutoff")
        _panel(ax, "abcd"[i], cond.replace("_", " "))
    lim = max(abs(v) for ax in axes[0] for v in ax.get_xlim())
    for ax in axes[0]:
        ax.set_xlim(-lim, lim)
    _legend(fig, [_h("o", ORANGE, "raw gap", markersize=6),
                  _h("s", BLUE, "gap net of TF-IDF's gap on the same reports", markersize=6)], 2)
    _stamp(fig, res)
    _save(fig, path, bottom=0.13)
    return True


# ---------------------------------------------------------------- figure: calibration

def reliability(res: dict, path: Path):
    panels = [(n, m["main"]["no_retrieval"]["all"]) for n, m in res["models"].items()
              if "reliability_bins" in m.get("main", {}).get("no_retrieval", {}).get("all", {})]
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(2.35 * len(panels), 3.1), sharey=True,
                             squeeze=False)
    for i, (ax, (name, a)) in enumerate(zip(axes[0], panels)):
        ax.plot([0, 1], [0, 1], color=RULE, linewidth=0.9, linestyle=(0, (3, 2)))
        ax.fill_between([0, 1], [0, 1], 0, color=ORANGE, alpha=0.06, linewidth=0)
        bins = [b for b in a["reliability_bins"] if b["count"]]
        xs, ys = [b["mean_conf"] for b in bins], [b["accuracy"] for b in bins]
        ax.plot(xs, ys, color=BLUE, linewidth=1.4)
        ax.scatter(xs, ys, s=[14 + 160 * b["count"] / a["n"] for b in bins], color=BLUE,
                   edgecolor=PAPER, linewidth=1, zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=3, colors=INK2)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.text(0.04, 0.96, f"ECE {a['ece']:.2f}\nn = {a['n']}", transform=ax.transAxes,
                fontsize=8, color=INK, va="top")
        ax.set_xlabel("stated confidence")
        _panel(ax, "abcd"[i], SHORT.get(name, name))
    axes[0][0].set_ylabel("fraction correct")
    _stamp(fig, res)
    _save(fig, path, bottom=0.03, top=0.93)
    return True


# --------------------------------------------------------------------- figure: probes

def probes(res: dict, path: Path):
    panels = [(n, m) for n, m in res["models"].items()
              if any("answer_change_rate" in p for p in (m.get("probes") or {}).values())]
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels) + 0.9, 2.6), sharey=True,
                             squeeze=False)
    xmax = 0.5
    for i, (ax, (name, m)) in enumerate(zip(axes[0], panels)):
        nf = m.get("noise_floor", {}).get("pairwise_disagreement")
        if nf:
            ax.axvspan(*nf["ci95"], color=GRID, zorder=0)
            ax.axvline(nf["estimate"], color=INK2, linewidth=0.9)
            ax.text(nf["estimate"], 1.0, " noise floor", transform=ax.get_xaxis_transform(),
                    fontsize=7, color=INK2, fontstyle="italic", va="bottom")
        items = [(p, v) for p, v in m["probes"].items() if "answer_change_rate" in v]
        for yi, (p, v) in enumerate(items[::-1]):
            c = v["answer_change_rate"]
            xmax = max(xmax, c["ci95"][1] + 0.1)
            ax.plot(c["ci95"], [yi, yi], color=BLUE, linewidth=1.6, solid_capstyle="butt")
            ax.plot(c["estimate"], yi, "o", color=BLUE, markersize=6, markeredgecolor=PAPER,
                    markeredgewidth=1.2, zorder=3)
        ax.set_yticks(range(len(items)), [f"{p}  (n={v['n']})" for p, v in items[::-1]])
        _axes(ax)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("answers changed vs unprobed prompt")
        _panel(ax, "abcd"[i], SHORT.get(name, name))
    for ax in axes[0]:
        ax.set_xlim(0, xmax)
    _stamp(fig, res)
    _save(fig, path, top=0.9)
    return True


# ------------------------------------------------------------------- figure: progress

def progress(done: dict, planned: dict, path: Path):
    names = [n for n in planned if planned[n]]
    if not names:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(names) + 0.7))
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        frac = min(done.get(n, 0) / planned[n], 1.0)
        ax.barh(yi, 1, height=0.42, color=GRID)
        ax.barh(yi, frac, height=0.42, color=BLUE)
        ax.text(1.015, yi, f"{done.get(n, 0)} / {planned[n]}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [SHORT.get(n, n) for n in names])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _axes(ax, xgrid=False)
    ax.set_xlabel("planned jobs finished")
    _save(fig, path, top=1.0)
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
