"""Generate 4 publication-grade figures (300 DPI PNG + vector SVG) directly from
pilot/report_master_final.json and verified classical baseline outputs.

Runs strictly locally at $0.00 API cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "pilot" / "report_master_final.json"
FIG_DIR = ROOT / "figures"

# Colorblind-safe Okabe-Ito palette
C_BLUE = "#0072B2"
C_VERM = "#D55E00"
C_TEAL = "#009E73"
C_AMBER = "#E69F00"
C_PURPLE = "#CC79A7"
C_SKY = "#56B4E9"
C_GRAY = "#5C646C"
C_DARK = "#1F2933"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
            "legend.fontsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444B54",
            "axes.linewidth": 0.9,
            "grid.color": "#CBD2D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.55,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def load_report() -> Dict[str, Any]:
    with REPORT_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_exp1(
    cells: List[Dict[str, Any]],
    model: str,
    family: str,
    track: str,
    regime: str,
    mode: str,
    substrate: str,
) -> Dict[str, Any]:
    for c in cells:
        if (
            c["model"] == model
            and c["family"] == family
            and c["track"] == track
            and c["regime"] == regime
            and c["mode"] == mode
            and c["substrate"] == substrate
        ):
            return c
    raise KeyError((model, family, track, regime, mode, substrate))


def find_exp2(
    comps: List[Dict[str, Any]],
    model: str,
    effort: str,
    task_type: str,
    track: str,
    variant: str,
) -> Dict[str, Any]:
    for c in comps:
        if (
            c["model"] == model
            and c["effort"] == effort
            and c.get("task_type", "majority3") == task_type
            and c["budget_track"] == track
            and c["variant"] == variant
        ):
            return c
    raise KeyError((model, effort, task_type, track, variant))


def p_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ==============================================================================
# FIGURE 1: Exp 1 — VBB Hard Wall vs Transparent & 4-Substrate Hierarchy
# ==============================================================================
def make_fig1(data: Dict[str, Any]) -> None:
    cells = data["exp1"]["cells"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), gridspec_kw={"width_ratios": [1.18, 1.0]})

    # --- Panel (a): VBB-Hard (n=160, m=1200, eta=0.22) vs Transparent (n=80, m=2200, eta=0.15) ---
    models_a = [
        ("Canon+Kikuchi-2\n(Classical Ref.)", None, None),
        ("Haiku 5.5", "claude-haiku-5-5", "4xor"),
        ("Sonnet 5.5", "claude-sonnet-5-5", "4xor_strict"),
        ("Opus 5.5", "claude-opus-5-5", "4xor_strict"),
        ("Fable 5.2", "claude-fable-5-2", "4xor_strict"),
    ]
    x = np.arange(len(models_a))
    w = 0.19

    # Verified classical reference values from baselines.py audit (N=24 each):
    # transparent: cv_adv = +1.000, vbb_hard: cv_adv = +0.000
    series_defs = [
        ("Transparent · code_clean", "transparent", "code_clean", C_BLUE, "", 1.000),
        ("Transparent · relu_weights", "transparent", "relu_weights", C_TEAL, "", 1.000),
        ("VBB-Hard · code_clean", "vbb_hard", "code_clean", C_VERM, "///", 0.000),
        ("VBB-Hard · relu_weights", "vbb_hard", "relu_weights", C_AMBER, "///", 0.000),
    ]

    ax1.axhspan(-0.25, 0.25, color="#F4F6F8", zorder=0, label="_nolegend_")
    ax1.axhline(0.0, color=C_DARK, linewidth=1.0, linestyle="--", zorder=1)

    for s_idx, (label, reg, sub, color, hatch, classic_val) in enumerate(series_defs):
        offsets = (s_idx - 1.5) * w
        vals, err_lo, err_hi, ps = [], [], [], []
        for _, m_id, trans_fam in models_a:
            if m_id is None:
                vals.append(classic_val)
                err_lo.append(0.0)
                err_hi.append(0.0)
                ps.append(1.0)
            else:
                fam = trans_fam if reg == "transparent" else "4xor_strict"
                c = find_exp1(cells, m_id, fam, "tool_python", reg, "WB", sub)
                v = c["adv"]
                vals.append(v)
                err_lo.append(max(0.0, v - c["ci_lo"]))
                err_hi.append(max(0.0, c["ci_hi"] - v))
                ps.append(c["p_value"])

        bars = ax1.bar(
            x + offsets,
            vals,
            width=w * 0.92,
            color=color,
            edgecolor="white" if not hatch else "#333333",
            linewidth=0.7,
            hatch=hatch,
            alpha=0.92,
            label=label,
            zorder=3,
        )
        # Draw CIs for LLM bars only (indices 1..4)
        ax1.errorbar(
            x[1:] + offsets,
            vals[1:],
            yerr=[err_lo[1:], err_hi[1:]],
            fmt="none",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=2.5,
            capthick=0.9,
            zorder=4,
        )
        # Significance markers on statistically significant transparent bars
        for i in range(1, len(models_a)):
            st = p_stars(ps[i])
            if st and vals[i] > 0.35:
                y_pos = min(1.06, vals[i] + err_hi[i] + 0.025)
                ax1.text(
                    x[i] + offsets,
                    y_pos,
                    st,
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    color=color,
                    zorder=5,
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels([m[0] for m in models_a])
    ax1.set_ylim(-0.72, 1.32)
    ax1.set_ylabel("Distinguishing Advantage  (Adv = TPR − FPR)")
    ax1.set_title("(a) Hardened 4-XOR VBB Barrier vs. Transparent Regime")
    ax1.grid(axis="y", zorder=0)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=2, frameon=True, framealpha=0.95, edgecolor="#D0D7DE")
    ax1.text(
        2.15,
        -0.635,
        "VBB-Hard (n=160, m=1200, η=0.22): ALL 4 frontier tiers blocked (all p_holm = 1.00)",
        ha="center",
        va="center",
        fontsize=8.2,
        style="italic",
        color="#8B2500",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F0", edgecolor="#E6B8A2", linewidth=0.7),
    )

    # --- Panel (b): 4-Substrate Hierarchy on Transparent 4-XOR (n=80, m=2200, eta=0.15) ---
    subs_order = ["code_clean", "code_obf", "relu_weights", "relu_weights_dense", "blackbox"]
    sub_labels = ["code_clean\n(Python)", "code_obf\n(MBA)", "relu_weights\n(2-layer MLP)", "relu_dense\n(3-layer W0)", "Black-Box\n(Oracle only)"]
    sx = np.arange(len(subs_order))

    ax2.axhline(0.0, color=C_DARK, linewidth=1.0, linestyle="--", zorder=1)
    ax2.grid(axis="y", zorder=0)

    model_lines = [
        ("Fable 5.2 (tool_python)", "claude-fable-5-2", "4xor_strict", "tool_python", C_PURPLE, "D", 2.3, -0.09),
        ("Sonnet 5.5 (tool_python)", "claude-sonnet-5-5", "4xor_strict", "tool_python", C_BLUE, "o", 2.1, -0.03),
        ("Haiku 5.5 (tool_python)", "claude-haiku-5-5", "4xor", "tool_python", C_TEAL, "s", 1.9, +0.03),
        ("Opus 5.5 (tool_python)", "claude-opus-5-5", "4xor_strict", "tool_python", C_AMBER, "^", 1.9, +0.09),
        ("Haiku 5.5 (tool_free)", "claude-haiku-5-5", "4xor_strict", "tool_free", C_GRAY, "x", 1.4, 0.0),
    ]

    for lbl, m_id, fam, trk, col, mk, lw, dx in model_lines:
        xs_m, ys_m, lo_m, hi_m = [], [], [], []
        for idx_s, sub in enumerate(subs_order):
            mode = "BB" if sub == "blackbox" else "WB"
            try:
                c = find_exp1(cells, m_id, fam, trk, "transparent", mode, sub)
            except KeyError:
                continue
            xs_m.append(sx[idx_s] + dx)
            ys_m.append(c["adv"])
            lo_m.append(max(0.0, c["adv"] - c["ci_lo"]))
            hi_m.append(max(0.0, c["ci_hi"] - c["adv"]))

        ls = "--" if trk == "tool_free" else "-"
        ax2.plot(xs_m, ys_m, color=col, marker=mk, markersize=6.5, linewidth=lw, linestyle=ls, label=lbl, zorder=4)
        if trk != "tool_free":
            ax2.errorbar(
                xs_m,
                ys_m,
                yerr=[lo_m, hi_m],
                fmt="none",
                ecolor=col,
                alpha=0.55,
                elinewidth=1.0,
                capsize=2.5,
                zorder=3,
            )

    ax2.annotate(
        "Fable 5.2 decompiles\n3-layer dense W0\n(+0.636, p=0.0045)",
        xy=(3 - 0.09, 0.636),
        xytext=(2.95, 0.96),
        fontsize=8.2,
        fontweight="bold",
        color="#7A2E63",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDF2F8", edgecolor=C_PURPLE, linewidth=0.8),
        arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.2, connectionstyle="arc3,rad=-0.15"),
        zorder=6,
    )

    ax2.set_xticks(sx)
    ax2.set_xticklabels(sub_labels)
    ax2.set_ylim(-0.62, 1.24)
    ax2.set_ylabel("Distinguishing Advantage  (Adv)")
    ax2.set_title("(b) 4-Substrate Observer-Relative Hierarchy (Transparent)")
    ax2.legend(loc="lower left", frameon=True, framealpha=0.95, edgecolor="#D0D7DE", fontsize=8.3)

    fig.tight_layout(w_pad=2.2)
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"fig1_exp1_vbb_and_substrates.{ext}")
    plt.close(fig)


# ==============================================================================
# FIGURE 2: Exp 2 — 9-Model Chronological Evolution (2025-09 -> 2026-11)
# ==============================================================================
def make_fig2(data: Dict[str, Any]) -> None:
    comps = data["exp2"]["comparisons"]
    models_9 = [
        ("claude-sonnet-4-5-20250929", "default", "Sonnet 4.5\n(2025-09)"),
        ("claude-haiku-4-5-20251001", "default", "Haiku 4.5\n(2025-10)"),
        ("claude-opus-4-6", "medium", "Opus 4.6\n(2026-02)"),
        ("claude-sonnet-4-6", "medium", "Sonnet 4.6*\n(2026-02)"),
        ("claude-opus-4-8", "medium", "Opus 4.8\n(2026-05)"),
        ("claude-haiku-5-5", "medium", "Haiku 5.5\n(2026-08)"),
        ("claude-sonnet-5-5", "medium", "Sonnet 5.5\n(2026-08)"),
        ("claude-opus-5-5", "medium", "Opus 5.5\n(2026-08)"),
        ("claude-fable-5-2", "medium", "Fable 5.2\n(2026-11)"),
    ]
    x = np.arange(len(models_9))

    fig, ax = plt.subplots(figsize=(13.2, 5.8))

    # Stage background bands
    ax.axvspan(-0.5, 1.5, color="#FDECEC", alpha=0.55, zorder=0)
    ax.axvspan(1.5, 6.5, color="#FFF8E6", alpha=0.55, zorder=0)
    ax.axvspan(6.5, 8.5, color="#ECFDF3", alpha=0.55, zorder=0)

    # Subtle negative inversion zone indicator
    ax.axhline(0.0, color=C_DARK, linewidth=1.3, zorder=2)
    ax.grid(axis="y", zorder=1)

    # Stage labels at top
    stage_y = 0.515
    ax.text(0.5, stage_y, "STAGE I\nTotal Inversion", ha="center", va="top", fontsize=9.0, fontweight="bold", color="#9B1C1C")
    ax.text(4.0, stage_y, "STAGE II: Selective VBB & Transparency Illusion\n(Opaque ReLU MLP solved → Readable Python still inverts)", ha="center", va="top", fontsize=9.0, fontweight="bold", color="#92400E")
    ax.text(7.5, stage_y, "STAGE III\nMetacognitive Escape", ha="center", va="top", fontsize=9.0, fontweight="bold", color="#065F46")

    variants = [
        ("wb_clean", "wb_clean (positive control)", C_TEAL, "^", "--", 1.5, -0.15),
        ("wb_decoy_trivial_guard", "wb_decoy_trivial_guard (dead-code control)", C_GRAY, "s", ":", 1.6, -0.05),
        ("wb_inv_relu", "wb_inv_relu (opaque 3-layer ReLU MLP)", C_BLUE, "D", "-", 2.0, +0.05),
        ("wb_inv_code", "wb_inv_code (anti-transparent Python)", C_VERM, "o", "-", 2.5, +0.15),
    ]

    for var, lbl, col, mk, ls, lw, dx in variants:
        ys, lo, hi, ps, decoys = [], [], [], [], []
        for m_id, eff, _ in models_9:
            c = find_exp2(comps, m_id, eff, "majority3", "matched_medium_budget", var)
            d = c["delta_wb"]
            ys.append(d)
            lo.append(max(0.0, d - c["ci_lo"]))
            hi.append(max(0.0, c["ci_hi"] - d))
            ps.append(c["p_value"])
            decoys.append(c["decoy_trap_rate"])

        mfc = "white" if var == "wb_decoy_trivial_guard" else col
        ax.plot(
            x + dx,
            ys,
            color=col,
            marker=mk,
            markerfacecolor=mfc,
            markeredgewidth=1.4,
            markersize=7.2 if var == "wb_inv_code" else 6.0,
            linewidth=lw,
            linestyle=ls,
            label=lbl,
            zorder=5 if var == "wb_inv_code" else 4,
        )
        ax.errorbar(
            x + dx,
            ys,
            yerr=[lo, hi],
            fmt="none",
            ecolor=col,
            alpha=0.7,
            elinewidth=1.1,
            capsize=2.6,
            zorder=3,
        )

        # Annotate key negative wb_inv_code points cleanly below error bars
        if var == "wb_inv_code":
            for i, (val, err_l, p_v, dec) in enumerate(zip(ys, lo, ps, decoys)):
                if val <= -0.07:
                    stars = p_stars(p_v)
                    y_txt = val - err_l - 0.022
                    # Slight stagger for crowded Haiku 4.5 / Sonnet 4.5
                    if i == 0:
                        y_txt -= 0.025
                    ax.text(
                        x[i] + dx,
                        y_txt,
                        f"{val:+.3f}{stars}\n({int(round(dec*100))}% dec.)",
                        ha="center",
                        va="top",
                        fontsize=7.8,
                        fontweight="bold",
                        color="#9E3A00",
                        zorder=6,
                    )

    ax.text(
        -0.42,
        -0.02,
        "Blackwell Inversion Zone (Δ_WB < 0)",
        ha="left",
        va="top",
        fontsize=8.3,
        style="italic",
        color="#7F1D1D",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([m[2] for m in models_9])
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.59, 0.545)
    ax.set_ylabel("White-Box Utility Gain  Δ_WB = Acc(WB) − Acc(BB)")
    ax.set_title("Chronological Evolution of Computational Blackwell Inversion (9 Frontier Models, majority3, 5 Turns, N=25)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.96, edgecolor="#D0D7DE", ncol=2, fontsize=8.6)

    fig.text(
        0.01,
        -0.02,
        "*Note: Sonnet 4.6 had an anomalously weak black-box baseline (BB = 0.689 vs. 0.88–1.00 for peers), producing positive Δ_WB despite a 12% decoy trap rate on wb_inv_code. Error bars: 95% paired bootstrap CI (5,000 reps).",
        fontsize=8.0,
        color=C_GRAY,
        style="italic",
    )

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"fig2_exp2_9model_chronology.{ext}")
    plt.close(fig)


# ==============================================================================
# FIGURE 3: Effort Double Dissociation & Theorem 1 Budget-Phase Transition
# ==============================================================================
def make_fig3(data: Dict[str, Any]) -> None:
    comps = data["exp2"]["comparisons"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2))

    # --- Panel (a): Double Dissociation of Reasoning Effort (low -> medium -> high, 5 turns) ---
    efforts = ["low", "medium", "high"]
    ex = np.arange(len(efforts))

    ax1.axhline(0.0, color=C_DARK, linewidth=1.1, linestyle="--", zorder=1)
    ax1.grid(axis="y", zorder=0)

    effort_series = [
        ("Haiku 5.5 · wb_inv_code (Overthinking Paradox)", "claude-haiku-5-5", "wb_inv_code", C_VERM, "o", "-", 2.4, -0.07),
        ("Haiku 5.5 · trivial_guard (Dead-code control)", "claude-haiku-5-5", "wb_decoy_trivial_guard", C_TEAL, "s", "--", 1.8, -0.02),
        ("Sonnet 5.5 · wb_inv_code (Partial mitigation)", "claude-sonnet-5-5", "wb_inv_code", C_BLUE, "D", "-", 2.4, +0.05),
        ("Sonnet 5.5 · wb_inv_relu (Opaque MLP)", "claude-sonnet-5-5", "wb_inv_relu", C_PURPLE, "^", ":", 1.8, +0.10),
    ]

    for lbl, m_id, var, col, mk, ls, lw, dx in effort_series:
        ys, lo, hi, decs = [], [], [], []
        for eff in efforts:
            c = find_exp2(comps, m_id, eff, "majority3", "matched_medium_budget", var)
            d = c["delta_wb"]
            ys.append(d)
            lo.append(max(0.0, d - c["ci_lo"]))
            hi.append(max(0.0, c["ci_hi"] - d))
            decs.append(c["decoy_trap_rate"])

        ax1.plot(ex + dx, ys, color=col, marker=mk, markersize=7.0, linewidth=lw, linestyle=ls, label=lbl, zorder=4)
        ax1.errorbar(ex + dx, ys, yerr=[lo, hi], fmt="none", ecolor=col, alpha=0.65, elinewidth=1.1, capsize=2.8, zorder=3)

        # Annotate decoy trap % cleanly off vertical error-bar stems
        if var == "wb_inv_code":
            for i, (y_v, dec_v) in enumerate(zip(ys, decs)):
                if m_id == "claude-haiku-5-5":
                    offsets_h = [(0.08, 0.045, "left"), (-0.09, -0.042, "right"), (-0.09, -0.030, "right")]
                    ox, oy, ha_s = offsets_h[i]
                    ax1.text(
                        ex[i] + dx + ox,
                        y_v + oy,
                        f"{int(round(dec_v*100))}% dec.",
                        ha=ha_s,
                        va="center",
                        fontsize=8.1,
                        fontweight="bold",
                        color="#9E3A00",
                    )
                else:
                    offsets_s = [(0.08, -0.015, "left"), (0.08, -0.025, "left"), (0.07, 0.025, "left")]
                    ox, oy, ha_s = offsets_s[i]
                    ax1.text(
                        ex[i] + dx + ox,
                        y_v + oy,
                        f"{int(round(dec_v*100))}% dec.",
                        ha=ha_s,
                        va="center",
                        fontsize=8.1,
                        fontweight="bold",
                        color="#004C78",
                    )

    ax1.set_xticks(ex)
    ax1.set_xticklabels(["Low Effort\n(Minimal thinking)", "Medium Effort\n(Standard)", "High Effort\n(Deep thinking)"])
    ax1.set_xlim(-0.32, 2.48)
    ax1.set_ylim(-0.46, 0.28)
    ax1.set_ylabel("White-Box Utility Gain  Δ_WB")
    ax1.set_title("(a) Double Dissociation of Reasoning Effort (5 Turns)")
    ax1.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#D0D7DE", fontsize=8.0)

    # --- Panel (b): Theorem 1 Budget-Phase Transition Delta_WB(B) (Haiku 5.5, medium) ---
    tracks = [
        ("matched_total_budget", "3 Turns\n(Tight Budget)"),
        ("matched_medium_budget", "5 Turns\n(Medium Budget)"),
        ("matched_query_free_code", "8 Turns\n(Generous Budget)"),
    ]
    bx = np.arange(len(tracks))

    ax2.axvspan(-0.4, 1.35, color="#FFF5F0", alpha=0.55, zorder=0)
    ax2.axvspan(1.35, 2.4, color="#ECFDF3", alpha=0.55, zorder=0)
    ax2.axhline(0.0, color=C_DARK, linewidth=1.1, linestyle="--", zorder=1)
    ax2.grid(axis="y", zorder=0)

    budget_series = [
        ("wb_clean (positive control)", "wb_clean", C_TEAL, "^", "--", 1.7, -0.07),
        ("wb_decoy_trivial_guard (dead-code control)", "wb_decoy_trivial_guard", C_GRAY, "s", ":", 1.8, -0.02),
        ("wb_inv_relu (opaque 3-layer MLP)", "wb_inv_relu", C_BLUE, "D", "-", 2.2, +0.03),
        ("wb_inv_code (anti-transparent Python)", "wb_inv_code", C_VERM, "o", "-", 2.5, +0.08),
    ]

    for lbl, var, col, mk, ls, lw, dx in budget_series:
        ys, lo, hi = [], [], []
        for trk, _ in tracks:
            c = find_exp2(comps, "claude-haiku-5-5", "medium", "majority3", trk, var)
            d = c["delta_wb"]
            ys.append(d)
            lo.append(max(0.0, d - c["ci_lo"]))
            hi.append(max(0.0, c["ci_hi"] - d))

        mfc = "white" if var == "wb_decoy_trivial_guard" else col
        ax2.plot(bx + dx, ys, color=col, marker=mk, markerfacecolor=mfc, markeredgewidth=1.3, markersize=7.0, linewidth=lw, linestyle=ls, label=lbl, zorder=4)
        ax2.errorbar(bx + dx, ys, yerr=[lo, hi], fmt="none", ecolor=col, alpha=0.65, elinewidth=1.1, capsize=2.8, zorder=3)

    ax2.annotate(
        "Tight budget (3t):\nBoth Python (−0.270***)\n& MLP (−0.274***) invert",
        xy=(0.08, -0.278),
        xytext=(0.30, -0.445),
        fontsize=7.8,
        fontweight="bold",
        color="#8B2500",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF5F0", edgecolor=C_VERM, linewidth=0.8),
        arrowprops=dict(arrowstyle="->", color=C_VERM, lw=1.1, connectionstyle="arc3,rad=0.18"),
        zorder=6,
    )
    ax2.annotate(
        "Generous budget (8t):\nAll variants recover\n(Δ_WB = +0.187)",
        xy=(1.94, 0.190),
        xytext=(1.32, 0.315),
        fontsize=8.0,
        fontweight="bold",
        color="#065F46",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#ECFDF3", edgecolor=C_TEAL, linewidth=0.8),
        arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.1, connectionstyle="arc3,rad=0.10"),
        zorder=6,
    )

    ax2.set_xticks(bx)
    ax2.set_xticklabels([t[1] for t in tracks])
    ax2.set_xlim(-0.45, 2.45)
    ax2.set_ylim(-0.54, 0.40)
    ax2.set_ylabel("White-Box Utility Gain  Δ_WB")
    ax2.set_title("(b) Theorem 1 Budget-Phase Curve Δ_WB(B) (Haiku 5.5)")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#D0D7DE", fontsize=8.0)

    fig.tight_layout(w_pad=2.2)
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"fig3_effort_budget_dissociation.{ext}")
    plt.close(fig)


# ==============================================================================
# FIGURE 4: Transparency Illusion & Cross-Task Generalization (majority3 vs dlist3)
# ==============================================================================
def make_fig4(data: Dict[str, Any]) -> None:
    comps = data["exp2"]["comparisons"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), gridspec_kw={"width_ratios": [1.18, 1.0]})

    groups = [
        ("Opus 4.6\n(majority3)", "claude-opus-4-6", "medium", "majority3", "p=0.0002"),
        ("Haiku 5.5\n(majority3)", "claude-haiku-5-5", "medium", "majority3", "p=0.017"),
        ("Sonnet 5.5\n(majority3)", "claude-sonnet-5-5", "medium", "majority3", "p=0.001"),
        ("Haiku 5.5\n(dlist3)", "claude-haiku-5-5", "medium", "dlist3", "p=0.005"),
        ("Sonnet 5.5\n(dlist3)", "claude-sonnet-5-5", "medium", "dlist3", "p=0.0004"),
    ]
    gx = np.arange(len(groups))
    w = 0.25

    # --- Panel (a): Delta_WB across trivial_guard, wb_inv_code, wb_inv_relu ---
    ax1.axhline(0.0, color=C_DARK, linewidth=1.1, linestyle="--", zorder=1)
    ax1.grid(axis="y", zorder=0)

    bar_defs = [
        ("wb_decoy_trivial_guard (dead code)", "wb_decoy_trivial_guard", C_GRAY, ""),
        ("wb_inv_code (readable Python)", "wb_inv_code", C_VERM, ""),
        ("wb_inv_relu (opaque 3-layer MLP)", "wb_inv_relu", C_BLUE, ""),
    ]

    ic_vals, ir_vals = [], []
    for idx_b, (lbl, var, col, hatch) in enumerate(bar_defs):
        off = (idx_b - 1.0) * w
        vals, lo, hi = [], [], []
        for _, m_id, eff, tt, _ in groups:
            c = find_exp2(comps, m_id, eff, tt, "matched_medium_budget", var)
            d = c["delta_wb"]
            vals.append(d)
            lo.append(max(0.0, d - c["ci_lo"]))
            hi.append(max(0.0, c["ci_hi"] - d))
        if var == "wb_inv_code":
            ic_vals = vals
        elif var == "wb_inv_relu":
            ir_vals = vals

        ax1.bar(gx + off, vals, width=w * 0.90, color=col, edgecolor="white", linewidth=0.7, alpha=0.92, label=lbl, zorder=3)
        ax1.errorbar(gx + off, vals, yerr=[lo, hi], fmt="none", ecolor="#222222", elinewidth=1.0, capsize=2.5, zorder=4)

    # Annotate exact Transparency Illusion gap (wb_inv_code - wb_inv_relu) above each cluster
    for i, (_, _, _, _, p_txt) in enumerate(groups):
        gap = ic_vals[i] - ir_vals[i]
        top_y = max(ir_vals[i], 0.03) + 0.095
        ax1.text(
            gx[i] + 0.12,
            top_y,
            f"Δ={gap:+.2f}\n({p_txt})",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#9E3A00",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFF5F0", edgecolor="#E6B8A2", linewidth=0.6),
            zorder=5,
        )

    ax1.axvline(2.5, color="#9AA5B1", linestyle=":", linewidth=1.2)
    ax1.text(1.0, -0.46, "Task 1: Sparse Majority-3", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=C_GRAY)
    ax1.text(3.5, -0.46, "Task 2: Rivest Decision List-3", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=C_GRAY)

    ax1.set_xticks(gx)
    ax1.set_xticklabels([g[0] for g in groups])
    ax1.set_ylim(-0.49, 0.38)
    ax1.set_ylabel("White-Box Utility Gain  Δ_WB")
    ax1.set_title("(a) Transparency Illusion & Cross-Task Replication (5 Turns)")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True, framealpha=0.95, edgecolor="#D0D7DE", fontsize=8.0)

    # --- Panel (b): Mechanism Proof — Decoy Trap Rate (%) + Equivalent Query Loss Gamma_WB ---
    ax2.grid(axis="y", zorder=0)
    for idx_b, (lbl, var, col, _) in enumerate(bar_defs):
        off = (idx_b - 1.0) * w
        rates = []
        gammas = []
        for _, m_id, eff, tt, _ in groups:
            c = find_exp2(comps, m_id, eff, tt, "matched_medium_budget", var)
            rates.append(100.0 * c["decoy_trap_rate"])
            gammas.append(c["gamma_wb_bits"])

        ax2.bar(gx + off, rates, width=w * 0.90, color=col, edgecolor="white", linewidth=0.7, alpha=0.92, label=lbl, zorder=3)
        for i, (r_val, g_val) in enumerate(zip(rates, gammas)):
            if var == "wb_inv_code":
                ax2.text(
                    gx[i] + off,
                    r_val + 1.8,
                    f"{int(round(r_val))}%\n({g_val:+.1f}b)",
                    ha="center",
                    va="bottom",
                    fontsize=7.6,
                    fontweight="bold",
                    color="#9E3A00",
                )
            elif var == "wb_inv_relu":
                ax2.text(
                    gx[i] + off,
                    2.0,
                    "0%",
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                    fontweight="bold",
                    color=C_BLUE,
                )

    ax2.axvline(2.5, color="#9AA5B1", linestyle=":", linewidth=1.2)
    ax2.set_xticks(gx)
    ax2.set_xticklabels([g[0] for g in groups])
    ax2.set_ylim(0, 92)
    ax2.set_ylabel("Decoy Anchoring Trap Rate  (%)")
    ax2.set_title("(b) Mechanism Proof: Decoy Trap Rate & Query Loss (Γ_WB bits)")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#D0D7DE", fontsize=8.1)

    fig.tight_layout(w_pad=2.2)
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"fig4_transparency_illusion_and_dlist3.{ext}")
    plt.close(fig)


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_report()
    make_fig1(data)
    make_fig2(data)
    make_fig3(data)
    make_fig4(data)
    print(f"Generated 4 figures (PNG + SVG) in {FIG_DIR}")


if __name__ == "__main__":
    main()
