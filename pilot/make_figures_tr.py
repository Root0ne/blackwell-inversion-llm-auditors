#!/usr/bin/env python3
"""
Elsevier (double-column 185 mm = 7.28 in) SCI-compliant Turkish figure generator
for: "Araç Destekli Büyük Dil Modeli Denetçilerinde Gözlemciye Göre Sanal Kara-Kutu
Özelliği ve Hesaplamalı Blackwell Tersinmesi"

Reads verified statistics from pilot/report_c3_unified.json (locked data) and, when it
exists, pilot/report_followup.json (follow-up runs). Zero API cost. Exports vector PDFs
(Type 42 fonts) + 600 DPI PNGs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C3_REPORT = ROOT / "pilot" / "report_c3_unified.json"
FOLLOWUP_REPORT = ROOT / "pilot" / "report_followup.json"
FIG_DIR = ROOT / "paper_tr" / "figures"

W_DOUBLE = 7.28

# Okabe-Ito colorblind-safe palette
C_BLUE = "#0072B2"
C_VERM = "#D55E00"
C_TEAL = "#009E73"
C_AMBER = "#E69F00"
C_PURPLE = "#CC79A7"
C_GRAY = "#5C646C"
C_DARK = "#1F2933"


def tr_num(val: float, decimals: int = 3, signed: bool = False) -> str:
    if abs(val) < 1e-12:
        val = 0.0
    txt = f"{val:+.{decimals}f}" if (signed and val != 0.0) else f"{val:.{decimals}f}"
    return txt.replace("-", "\u2212").replace(".", ",")


def tr_axis_fmt(decimals: int = 1, signed: bool = False) -> FuncFormatter:
    return FuncFormatter(lambda x, _pos: tr_num(x, decimals=decimals, signed=signed))


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
        "font.size": 7.8,
        "axes.labelsize": 7.9,
        "axes.labelcolor": C_DARK,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.9,
        "xtick.color": C_DARK,
        "ytick.color": C_DARK,
        "legend.fontsize": 6.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": C_DARK,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
        "hatch.linewidth": 0.55,
    })


def p_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def save_both(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")
    plt.close(fig)


def get_exp1_row(rows: List[Dict[str, Any]], model: str, regime: str, substrate: str, track: str = "tool_python") -> Dict[str, Any]:
    for r in rows:
        if r["model"] == model and r["regime"] == regime and r["substrate"] == substrate and r["track"] == track:
            return r
    raise KeyError((model, regime, substrate, track))


def get_exp2_cell(
    cells: List[Dict[str, Any]],
    model: str,
    effort: str = "medium",
    track: str = "matched_medium_budget",
    task: str = "majority3",
    mt: str = "4096",
) -> Dict[str, Any]:
    for ef_try in ([effort, "default"] if effort == "medium" else [effort]):
        for c in cells:
            if (
                c["model"] == model
                and c["effort"] == ef_try
                and c["budget_track"] == track
                and c["task_type"] == task
                and c["max_tokens"] == mt
            ):
                return c
    raise KeyError((model, effort, track, task, mt))


# ==============================================================================
# ŞEKİL 1 (fig0): Yalın ve Tek-Temalı Deneysel Mimari Akışı
# ==============================================================================
def make_fig0_workflow() -> None:
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.25))
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 3.10)
    ax.axis("off")

    # Single cohesive monochrome slate-navy theme
    PRIMARY = "#1E3A5F"
    TINT = "#F4F7FA"
    BORDER = "#94A3B8"
    INK = "#1E293B"
    MUTED = "#475569"

    text_bounds: List[tuple] = []

    boxes = [
        {
            "x": 0.06, "w": 2.12, "dashed": False,
            "num": "1", "title": "Test Üretimi",
            "items": [
                ("Deney 1 (Arka Kapı)", "4-XOR kod, MBA ve ReLU"),
                ("Deney 2 (Yanıltıcı Dal)", "5 beyaz-kutu varyantı"),
                ("Kontrol", "Sızıntı testi (η = 0,50)"),
            ],
        },
        {
            "x": 2.60, "w": 2.12, "dashed": True,
            "num": "2", "title": "Yalıtılmış Ortam",
            "items": [
                ("Sorgu Arayüzü", "Ortak BB/WB (q ≤ 80)"),
                ("Python Süreci", "Dosya yalnızca WB kipinde"),
                ("Eşit Bütçe", "Kip başına 3, 5 veya 8 tur"),
            ],
        },
        {
            "x": 5.26, "w": 2.12, "dashed": False,
            "num": "3", "title": "Değerlendirme",
            "items": [
                ("Kapalı Modeller", "Claude (9) ve Gemini (4)"),
                ("Açık Ağırlık", "Gemma 4 (26B ve 31B)"),
                ("Klasik Referans", "8 saldırı (Kikuchi-2, SA)"),
            ],
        },
        {
            "x": 7.80, "w": 2.14, "dashed": False,
            "num": "4", "title": "Analiz",
            "items": [
                ("İstatistik", "Fisher, permütasyon, Holm"),
                ("Kayıp Ayrıştırması", "Tuzak, kesilme, diğer"),
                ("Bütünlük", "7.631 deneme, 18 test"),
            ],
        },
    ]

    for idx, b in enumerate(boxes):
        x, w = b["x"], b["w"]
        y, h = 0.20, 2.70

        # Outer card (white fill, single theme stroke)
        card = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.015,rounding_size=0.09",
            facecolor="white",
            edgecolor=PRIMARY if b["dashed"] else BORDER,
            linewidth=1.15 if b["dashed"] else 0.95,
            linestyle="--" if b["dashed"] else "-",
            zorder=2,
        )
        ax.add_patch(card)

        # Clean top header strip in uniform theme tint
        hdr_h = 0.56
        hdr = FancyBboxPatch(
            (x, y + h - hdr_h), w, hdr_h,
            boxstyle="round,pad=0.015,rounding_size=0.09",
            facecolor=TINT, edgecolor="none", zorder=3,
        )
        ax.add_patch(hdr)
        # Subtle divider line under header
        ax.plot([x + 0.04, x + w - 0.04], [y + h - hdr_h, y + h - hdr_h],
                color=BORDER, linewidth=0.6, zorder=4)

        # Number badge (single navy accent)
        badge = FancyBboxPatch(
            (x + 0.11, y + h - 0.43), 0.28, 0.29,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor=PRIMARY, edgecolor="none", zorder=5,
        )
        ax.add_patch(badge)
        ax.text(
            x + 0.25, y + h - 0.285, b["num"],
            ha="center", va="center", fontsize=6.4, fontweight="bold",
            color="white", zorder=6,
        )
        t_hdr = ax.text(
            x + 0.47, y + h - 0.285, b["title"],
            ha="left", va="center", fontsize=6.8, fontweight="bold",
            color=PRIMARY, zorder=6,
        )
        text_bounds.append((t_hdr, x, y + h - hdr_h, x + w, y + h, f"HDR:{b['title']}"))

        # Clean evenly spaced content rows
        row_ys = [1.84, 1.18, 0.54]
        for (head, desc), ry in zip(b["items"], row_ys):
            if head:
                t1 = ax.text(
                    x + 0.13, ry + 0.13, head,
                    ha="left", va="center", fontsize=6.0, fontweight="bold",
                    color=INK, zorder=5,
                )
                t2 = ax.text(
                    x + 0.13, ry - 0.10, desc,
                    ha="left", va="center", fontsize=5.65, color=MUTED, zorder=5,
                )
                text_bounds.append((t1, x, y, x + w, y + h, f"H:{head}"))
                text_bounds.append((t2, x, y, x + w, y + h, f"D:{desc}"))
            else:
                # Continuation line tightly paired under previous item
                t2 = ax.text(
                    x + 0.13, ry + 0.30, desc,
                    ha="left", va="center", fontsize=5.65, color=MUTED, zorder=5,
                )
                text_bounds.append((t2, x, y, x + w, y + h, f"D:{desc}"))

    # Clean single-color connecting arrows between the 4 stages
    def draw_conn(x1: float, x2: float, style: str = "-|>", label: str = "") -> None:
        arr = FancyArrowPatch(
            (x1, 1.55), (x2, 1.55),
            arrowstyle=style, mutation_scale=8.5,
            linewidth=1.15, color=PRIMARY, zorder=7,
        )
        ax.add_patch(arr)
        if label:
            ax.text(
                (x1 + x2) / 2.0, 1.72, label,
                ha="center", va="bottom", fontsize=5.2, color=MUTED, linespacing=1.1,
            )

    draw_conn(2.22, 2.56)
    draw_conn(4.76, 5.22, style="<|-|>", label="Araç")
    draw_conn(7.42, 7.76)

    # Automated mathematical bounding-box overflow verification
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    overflows = []
    for t_obj, bx0, by0, bx1, by1, tag in text_bounds:
        bb = t_obj.get_window_extent(renderer=renderer).transformed(inv)
        if bb.x0 < bx0 + 0.02 or bb.x1 > bx1 - 0.04 or bb.y0 < by0 + 0.02 or bb.y1 > by1 - 0.02:
            overflows.append((tag, (round(bb.x0, 2), round(bb.x1, 2)), (round(bx0, 2), round(bx1, 2))))
    if overflows:
        raise RuntimeError(f"Figure 1 text overflow detected: {overflows}")

    save_both(fig, "fig0_workflow")


# ==============================================================================
# ŞEKİL 2 (fig1): Deney 1 — 4xor_strict (N=22-60) + 4xor_fixed120 m Faz Geçişi
# ==============================================================================
def make_fig1(c3: Dict[str, Any]) -> None:
    strict = c3["exp1_4xor_strict"]
    fixed120 = c3["exp1_4xor_fixed120"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(W_DOUBLE, 3.45),
        gridspec_kw={"width_ratios": [1.22, 1.0], "wspace": 0.34},
    )

    models_a = [
        ("Sonnet 5.5", "claude-sonnet-5-5"),
        ("Opus 5.5", "claude-opus-5-5"),
        ("Fable 5.2", "claude-fable-5-2"),
        ("Gemini 4\nCarbon", "gemini-4-carbon"),
    ]
    bars_spec = [
        ("Çözülebilir: sade kod", "transparent", "code_clean", C_BLUE, ""),
        ("Çözülebilir: yoğun ağırlık", "transparent", "relu_weights_dense", C_TEAL, "//"),
        ("Zor: sade kod", "vbb_hard", "code_clean", C_VERM, ""),
        ("Zor: yoğun ağırlık", "vbb_hard", "relu_weights_dense", C_AMBER, "xx"),
    ]
    x = np.arange(len(models_a))
    width = 0.19
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    ax1.axhline(0.0, color=C_DARK, linewidth=0.75, zorder=1)

    for b_idx, (lbl, reg, sub, col, hatch) in enumerate(bars_spec):
        vals, lo_e, hi_e, stars_l = [], [], [], []
        for _, mkey in models_a:
            r = get_exp1_row(strict, mkey, reg, sub)
            adv = r["adv"]
            vals.append(adv)
            lo_e.append(max(0.0, adv - r["ci_lo"]))
            hi_e.append(max(0.0, r["ci_hi"] - adv))
            stars_l.append(p_stars(r.get("p_holm_model", 1.0)))

        pos = x + offsets[b_idx]
        ax1.bar(
            pos, vals, width=width * 0.92,
            color=col, edgecolor=C_DARK, linewidth=0.55, hatch=hatch,
            label=lbl, zorder=3,
        )
        ax1.errorbar(
            pos, vals, yerr=[lo_e, hi_e], fmt="none",
            ecolor=C_DARK, elinewidth=0.7, capsize=2.0, capthick=0.65, zorder=4,
        )
        for xi, v, he, st in zip(pos, vals, hi_e, stars_l):
            if st:
                y_st = max(v, 0.0) + he + 0.03
                ax1.text(xi, y_st, st, ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=C_DARK, zorder=5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([m[0] for m in models_a])
    ax1.set_ylabel("Ayırt Etme Avantajı (Adv = TPR \u2212 FPR)")
    ax1.set_ylim(-0.52, 1.16)
    ax1.yaxis.set_major_formatter(tr_axis_fmt(1, signed=True))
    ax1.set_title("(a) Sertleştirilmiş 4-XOR Ailesi (N = 22\u201360)", loc="left", fontsize=7.5, fontweight="bold", pad=7)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=6.1, handlelength=1.4)

    # Panel (b): Single-variable m phase transition (4xor_fixed120, n=120, k=4, eta=0.20)
    regimes_b = ["m = 900 denklem\n(zor)", "m = 4.400 denklem\n(çözülebilir)"]
    xb = np.arange(len(regimes_b))
    series_b = [
        ("Klasik saldırı (Canon + Kikuchi-2)", [-0.04, 1.00], [0.0, 0.0], [0.0, 0.0], C_GRAY, "s", "--"),
        ("Haiku 5.5 (sade kod)", [], [], [], C_VERM, "o", "-"),
        ("Sonnet 5.5 (sade kod)", [], [], [], C_BLUE, "D", "-"),
        ("Haiku 5.5 (seyrek ağırlık)", [], [], [], C_AMBER, "^", ":"),
        ("Sonnet 5.5 (seyrek ağırlık)", [], [], [], C_TEAL, "v", ":"),
    ]
    model_sub_map = [
        None,
        ("claude-haiku-5-5", "code_clean"),
        ("claude-sonnet-5-5", "code_clean"),
        ("claude-haiku-5-5", "relu_weights"),
        ("claude-sonnet-5-5", "relu_weights"),
    ]
    for s_idx, ms in enumerate(model_sub_map):
        if ms is None:
            continue
        mkey, sub = ms
        for reg in ("vbb_hard", "transparent"):
            r = get_exp1_row(fixed120, mkey, reg, sub)
            series_b[s_idx][1].append(r["adv"])
            series_b[s_idx][2].append(max(0.0, r["adv"] - r["ci_lo"]))
            series_b[s_idx][3].append(max(0.0, r["ci_hi"] - r["adv"]))

    ax2.axhline(0.0, color=C_DARK, linewidth=0.75, zorder=1)
    x_jit = [-0.08, -0.04, 0.0, 0.04, 0.08]
    for idx, (lbl, vals, lo_e, hi_e, col, mk, ls) in enumerate(series_b):
        xp = xb + x_jit[idx]
        ax2.plot(xp, vals, color=col, marker=mk, linestyle=ls, linewidth=1.4, markersize=5.0, label=lbl, zorder=4)
        if any(e > 0 for e in lo_e + hi_e):
            ax2.errorbar(xp, vals, yerr=[lo_e, hi_e], fmt="none", ecolor=col, elinewidth=0.75, capsize=2.2, alpha=0.85, zorder=3)

    ax2.set_xticks(xb)
    ax2.set_xticklabels(regimes_b)
    ax2.set_xlim(-0.35, 1.35)
    ax2.set_ylim(-0.68, 1.16)
    ax2.yaxis.set_major_formatter(tr_axis_fmt(1, signed=True))
    ax2.set_ylabel("Ayırt Etme Avantajı (Adv)")
    ax2.set_title("(b) Sabit Boyutlu 4-XOR: Denklem Sayısının Etkisi", loc="left", fontsize=7.4, fontweight="bold", pad=7)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=5.9)

    save_both(fig, "fig1_exp1_vbb_hierarchy")


# ==============================================================================
# ŞEKİL (fig2): Dokuz Anthropic modelinde varyantlara göre Delta_WB (grup sırasıyla)
# ==============================================================================
def make_fig2(c3: Dict[str, Any], fu: Dict[str, Any]) -> None:
    cells = c3["exp2"]["cells"]
    # Metindeki üç grup sırası; grup içinde yayın tarihine göre.
    models = [
        ("claude-sonnet-4-5-20250929", "Sonnet\n4.5"),
        ("claude-haiku-4-5-20251001", "Haiku\n4.5"),
        ("claude-opus-4-6", "Opus\n4.6"),
        ("claude-sonnet-4-6", "Sonnet\n4.6*"),
        ("claude-opus-4-8", "Opus\n4.8"),
        ("claude-haiku-5-5", "Haiku\n5.5"),
        ("claude-sonnet-5-5", "Sonnet\n5.5"),
        ("claude-opus-5-5", "Opus\n5.5"),
        ("claude-fable-5-2", "Fable\n5.2"),
    ]
    groups = [("Birinci grup", 0, 1), ("İkinci grup", 2, 6), ("Üçüncü grup", 7, 8)]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.25))

    ax.axhspan(-0.52, 0.0, color="#FDF2EE", alpha=0.55, zorder=0)
    for i in range(0, len(models), 2):
        ax.axvspan(i - 0.5, i + 0.5, color="#000000", alpha=0.035, linewidth=0, zorder=0)
    ax.axhline(0.0, color=C_DARK, linewidth=0.9, zorder=2)

    fu_refut = {r["model"]: r["delta_wb_refutable"] for r in fu.get("part_a_refutable_extension", {}).get("models", [])}

    def refut_delta(mkey: str) -> Any:
        c = get_exp2_cell(cells, mkey)
        if "wb_decoy_long_refutable_guard" in c["variants"]:
            vd = c["variants"]["wb_decoy_long_refutable_guard"]["vs_bb"]
            return vd["delta_wb"], vd["ci_lo"], vd["ci_hi"]
        if mkey in fu_refut:
            d = fu_refut[mkey]
            return d["diff"], d["ci95"][0], d["ci95"][1]
        return None

    specs = [
        ("wb_clean", "Sade kod", C_TEAL, "^", 4.6, -0.28),
        ("wb_decoy_trivial_guard", "Kısa kontrol", C_GRAY, "s", 4.2, -0.14),
        ("wb_decoy_long_refutable_guard", "Çürütülebilir kontrol", C_PURPLE, "v", 4.8, 0.0),
        ("wb_inv_code", "Zor koruyuculu kod", C_VERM, "o", 5.4, +0.14),
        ("wb_inv_relu", "Opak ağırlık", C_BLUE, "D", 4.2, +0.28),
    ]
    for var, lbl, col, mk, ms, xoff in specs:
        xs, ys, lo_e, hi_e = [], [], [], []
        for i, (mkey, _) in enumerate(models):
            if var == "wb_decoy_long_refutable_guard":
                t = refut_delta(mkey)
                if t is None:
                    continue
                d, lo, hi = t
            else:
                vd = get_exp2_cell(cells, mkey)["variants"][var]["vs_bb"]
                d, lo, hi = vd["delta_wb"], vd["ci_lo"], vd["ci_hi"]
            xs.append(i + xoff)
            ys.append(d)
            lo_e.append(max(0.0, d - lo))
            hi_e.append(max(0.0, hi - d))
        hollow = var in ("wb_decoy_trivial_guard", "wb_decoy_long_refutable_guard")
        ax.errorbar(xs, ys, yerr=[lo_e, hi_e], fmt="none", ecolor=col, elinewidth=0.7, capsize=1.6, alpha=0.75, zorder=3)
        ax.plot(xs, ys, linestyle="none", marker=mk, markersize=ms, color=col,
                markerfacecolor="white" if hollow else col, markeredgecolor=col,
                markeredgewidth=1.2 if hollow else 0.6, label=lbl, zorder=5)

    i46 = [m[0] for m in models].index("claude-opus-4-6")
    c_o46_8k = get_exp2_cell(cells, "claude-opus-4-6", mt="8192")
    d_8k = c_o46_8k["variants"]["wb_inv_code"]["vs_bb"]["delta_wb"]
    ax.scatter([i46 + 0.14], [d_8k], marker="*", s=62, color=C_AMBER, edgecolor=C_DARK, linewidth=0.6,
               label="Opus 4.6, zor koruyuculu kod, token tavanı 8.192", zorder=7)

    for (glbl, i0, i1) in groups:
        ax.text((i0 + i1) / 2.0, 0.575, glbl, ha="center", va="center", fontsize=6.6, fontweight="bold", color=C_DARK)
    for xsep in (1.5, 6.5):
        ax.axvline(xsep, color=C_DARK, linestyle="-", linewidth=0.7, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in models])
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylabel(r"Net Beyaz-Kutu Avantajı ($\Delta_{\mathrm{WB}} = \mathrm{WB} - \mathrm{BB}$)")
    ax.set_ylim(-0.52, 0.62)
    ax.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    ax.yaxis.set_major_formatter(tr_axis_fmt(1, signed=True))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, facecolor="white", edgecolor="#D1D5DB", framealpha=0.95, fontsize=6.0)
    ax.text(0.995, 0.03, r"Tersinme ($\Delta_{\mathrm{WB}} < 0$)", transform=ax.transAxes,
            ha="right", fontsize=6.6, color=C_VERM, fontweight="bold")

    save_both(fig, "fig2_exp2_chronology")


# ==============================================================================
# ŞEKİL 4 (fig3): 4-Model Efor Taraması (Aşırı Düşünme) + Eşit 8t/8t Bütçe Fazı
# ==============================================================================
def make_fig3(c3: Dict[str, Any]) -> None:
    cells = c3["exp2"]["cells"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(W_DOUBLE, 3.45),
        gridspec_kw={"width_ratios": [1.04, 1.0], "wspace": 0.30},
    )

    efforts = ["low", "medium", "high"]
    eff_labels = ["Düşük\n(low)", "Orta\n(medium)", "Yüksek\n(high)"]
    xe = np.arange(len(efforts))

    effort_models = [
        ("Haiku 5.5", "claude-haiku-5-5", C_VERM, "o", "-", 1.8),
        ("Gemini 3.5 Flash-Lite", "gemini-3.5-flash-lite", C_AMBER, "s", "--", 1.7),
        ("Sonnet 5.5", "claude-sonnet-5-5", C_BLUE, "D", "-", 1.7),
        ("Opus 4.8", "claude-opus-4-8", C_TEAL, "^", ":", 1.5),
    ]
    for lbl, mkey, col, mk, ls, lw in effort_models:
        rates = [100.0 * get_exp2_cell(cells, mkey, effort=ef)["variants"]["wb_inv_code"]["decoy_rate"] for ef in efforts]
        ax1.plot(xe, rates, color=col, marker=mk, linestyle=ls, linewidth=lw, markersize=5.2, label=lbl, zorder=4)
        for xi, r in zip(xe, rates):
            if mkey == "claude-opus-4-8" and xi == 0:
                dy = -5.5
            elif mkey == "gemini-3.5-flash-lite" and xi == 1:
                dy = -6.0
            elif mkey == "claude-opus-4-8" and xi == 1:
                dy = +3.2
            else:
                dy = +3.0
            ax1.text(xi, r + dy, f"%{r:.0f}", ha="center", va="bottom" if dy > 0 else "top",
                     fontsize=6.3, fontweight="bold", color=col)

    ax1.set_xticks(xe)
    ax1.set_xticklabels(eff_labels)
    ax1.set_xlim(-0.32, 2.32)
    ax1.set_ylim(-2, 78)
    ax1.set_ylabel("Zor Koruyuculu Kodda Tuzak Oranı (%)")
    ax1.set_title("(a) Efor Taraması: İki Sağlayıcıdan Dört Model", loc="left", fontsize=7.2, fontweight="bold", pad=7)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=5.8)

    tracks_b = [
        ("3 tur", "matched_total_budget"),
        ("5 tur", "matched_medium_budget"),
        ("8 tur", "matched_wide_equal_budget"),
    ]
    xb = np.arange(len(tracks_b))
    ax2.axhline(0.0, color=C_DARK, linewidth=0.8, zorder=1)
    ax2.axhspan(-0.40, 0.0, color="#FDF2EE", alpha=0.5, zorder=0)

    b_series = [
        ("Haiku 5.5 · zor koruyuculu kod", "claude-haiku-5-5", "wb_inv_code", C_VERM, "o", "-", 1.7, -0.05),
        ("Haiku 5.5 · opak ağırlık", "claude-haiku-5-5", "wb_inv_relu", C_AMBER, "s", "--", 1.35, -0.02),
        ("Sonnet 5.5 · zor koruyuculu kod", "claude-sonnet-5-5", "wb_inv_code", C_BLUE, "D", "-", 1.7, +0.02),
        ("Sonnet 5.5 · opak ağırlık", "claude-sonnet-5-5", "wb_inv_relu", C_TEAL, "^", ":", 1.35, +0.05),
    ]
    for lbl, mkey, var, col, mk, ls, lw, xoff in b_series:
        ys, lo_e, hi_e = [], [], []
        for _, tkey in tracks_b:
            c = get_exp2_cell(cells, mkey, effort="medium", track=tkey)
            vd = c["variants"][var]["vs_bb"]
            d = vd["delta_wb"]
            ys.append(d)
            lo_e.append(max(0.0, d - vd["ci_lo"]))
            hi_e.append(max(0.0, vd["ci_hi"] - d))
        ax2.plot(xb + xoff, ys, color=col, marker=mk, linestyle=ls, linewidth=lw, markersize=4.8, label=lbl, zorder=4)
        ax2.errorbar(xb + xoff, ys, yerr=[lo_e, hi_e], fmt="none", ecolor=col, elinewidth=0.7, capsize=2.0, alpha=0.75, zorder=3)

    ax2.annotate(
        "8 turda iki modelde de\n$\\Delta_{\\mathrm{WB}} \\geq 0$ ve tuzak oranı %0",
        xy=(2.0, 0.0), xytext=(0.95, -0.33),
        fontsize=5.9, color=C_DARK, fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": C_DARK, "lw": 0.75},
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#ECFDF5", "edgecolor": C_TEAL, "lw": 0.75},
        zorder=6,
    )
    ax2.set_xticks(xb)
    ax2.set_xticklabels([t[0] for t in tracks_b])
    ax2.set_xlabel("Kip başına konuşma turu sayısı")
    ax2.set_xlim(-0.32, 2.32)
    ax2.set_ylim(-0.40, 0.28)
    ax2.set_ylabel(r"Net Beyaz-Kutu Avantajı ($\Delta_{\mathrm{WB}} = \mathrm{WB} - \mathrm{BB}$)")
    ax2.yaxis.set_major_formatter(tr_axis_fmt(1, signed=True))
    ax2.set_title("(b) Tur Bütçesinin Etkisi (BB ve WB Eşit Turda)", loc="left", fontsize=7.2, fontweight="bold", pad=7)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=5.8)

    save_both(fig, "fig3_exp2_effort_budget")


# ==============================================================================
# ŞEKİL 5 (fig4): B8 Nedensel Mekanizma + 3-Yönlü Kayıp Ayrıştırması
# ==============================================================================
def make_fig4(c3: Dict[str, Any]) -> None:
    cells = c3["exp2"]["cells"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(W_DOUBLE, 3.85),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.56},
    )

    rows_a = [
        ("Haiku 5.5 (3 tur)", "claude-haiku-5-5", "medium", "matched_total_budget", "majority3"),
        ("Haiku 5.5 (high efor)", "claude-haiku-5-5", "high", "matched_medium_budget", "majority3"),
        ("Sonnet 5.5", "claude-sonnet-5-5", "medium", "matched_medium_budget", "majority3"),
        ("Sonnet 5.5 (Karar Listesi-3)", "claude-sonnet-5-5", "medium", "matched_medium_budget", "dlist3"),
        ("Opus 4.6", "claude-opus-4-6", "medium", "matched_medium_budget", "majority3"),
        ("Gemini 3.8 Flash", "gemini-3.8-flash", "medium", "matched_medium_budget", "majority3"),
        ("Gemma 4 31B", "gemma-4-31b-it", "medium", "matched_medium_budget", "majority3"),
    ]
    y = np.arange(len(rows_a))
    h = 0.19
    v_specs = [
        ("wb_inv_relu", "Opak ağırlık", C_BLUE, "", +1.5 * h),
        ("wb_decoy_trivial_guard", "Kısa kontrol", C_GRAY, "", +0.5 * h),
        ("wb_decoy_long_refutable_guard", "Çürütülebilir kontrol", C_PURPLE, "//", -0.5 * h),
        ("wb_inv_code", "Zor koruyuculu kod", C_VERM, "", -1.5 * h),
    ]
    for vkey, lbl, col, hatch, yoff in v_specs:
        accs = [get_exp2_cell(cells, m, effort=ef, track=trk, task=tsk)["variants"][vkey]["acc"] for _, m, ef, trk, tsk in rows_a]
        ax1.barh(y + yoff, np.array(accs) - 0.5, left=0.5, height=h * 0.90, color=col, edgecolor=C_DARK, linewidth=0.45, hatch=hatch, label=lbl, zorder=3)

    ax1.set_yticks(y)
    ax1.set_yticklabels([r[0] for r in rows_a], fontsize=6.6)
    ax1.invert_yaxis()
    ax1.set_xlim(0.5, 1.03)
    ax1.xaxis.set_major_formatter(tr_axis_fmt(1, signed=False))
    ax1.set_xlabel("Beyaz-kutu test doğruluğu (Acc)")
    ax1.set_title("(a) Dört Beyaz-Kutu Varyantının Doğruluğu", loc="left", fontsize=7.0, fontweight="bold", pad=7)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.48, -0.17), ncol=2, frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=5.8)

    rows_b = [
        ("Sonnet 4.5", "claude-sonnet-4-5-20250929", "medium", "4096"),
        ("Opus 4.6 (tavan 4.096)", "claude-opus-4-6", "medium", "4096"),
        ("Opus 4.6 (tavan 8.192)", "claude-opus-4-6", "medium", "8192"),
        ("Haiku 5.5 (high efor)", "claude-haiku-5-5", "high", "4096"),
        ("Sonnet 5.5", "claude-sonnet-5-5", "medium", "4096"),
        ("Gemini 3.5 Flash-Lite\n(high efor)", "gemini-3.5-flash-lite", "high", "4096"),
        ("Gemma 4 31B", "gemma-4-31b-it", "medium", "4096"),
        ("Gemma 4 26B", "gemma-4-26b-a4b-it", "medium", "4096"),
    ]
    yb = np.arange(len(rows_b))
    ld, lt, lo = [], [], []
    for _, m, ef, mt in rows_b:
        c = get_exp2_cell(cells, m, effort=ef, mt=mt)
        v = c["variants"]["wb_inv_code"]
        ld.append(v["loss_decoy"])
        lt.append(v["loss_trunc"])
        lo.append(v["loss_other"])
    ld_a, lt_a, lo_a = np.array(ld), np.array(lt), np.array(lo)

    ax2.barh(yb, ld_a, height=0.62, color=C_VERM, edgecolor=C_DARK, linewidth=0.5, label="Tuzak kural seçildi")
    ax2.barh(yb, lt_a, left=ld_a, height=0.62, color=C_AMBER, edgecolor=C_DARK, linewidth=0.5, hatch="//", label="Token kesilmesi")
    ax2.barh(yb, lo_a, left=ld_a + lt_a, height=0.62, color=C_GRAY, edgecolor=C_DARK, linewidth=0.5, hatch="..", label="Diğer yanlış yanıt veya geçersiz biçim")

    for yi, tot in zip(yb, ld_a + lt_a + lo_a):
        ax2.text(tot + 0.012, yi, tr_num(tot, 2), va="center", ha="left", fontsize=6.2, color=C_DARK, fontweight="bold")

    ax2.set_yticks(yb)
    ax2.set_yticklabels([r[0] for r in rows_b], fontsize=6.6)
    ax2.invert_yaxis()
    ax2.set_xlim(0.0, 0.58)
    ax2.xaxis.set_major_formatter(tr_axis_fmt(1, signed=False))
    ax2.set_xlabel("Doğruluk kaybı (1 \u2212 Acc)")
    ax2.set_title("(b) Zor Koruyuculu Kodda Doğruluk Kaybının Bileşenleri", loc="left", fontsize=7.0, fontweight="bold", pad=7)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.48, -0.17), ncol=2, frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=5.8)

    save_both(fig, "fig4_transparency_illusion")


# ==============================================================================
# ŞEKİL (fig5): Google Gemini + Gemma 4 doğruluğu; 15 modelde tuzak oranları
# ==============================================================================
def make_fig5_cross_provider(c3: Dict[str, Any], fu: Dict[str, Any]) -> None:
    cells = c3["exp2"]["cells"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(W_DOUBLE, 4.35),
        gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.62},
    )

    gp_models = [
        ("Gemma 4\n26B", "gemma-4-26b-a4b-it"),
        ("Gemma 4\n31B", "gemma-4-31b-it"),
        ("Gemini 3.1\nFlash-Lite", "gemini-3.1-flash-lite"),
        ("Gemini 3.5\nFlash-Lite", "gemini-3.5-flash-lite"),
        ("Gemini 3.8\nFlash", "gemini-3.8-flash"),
        ("Gemini 4\nCarbon", "gemini-4-carbon"),
    ]
    xg = np.arange(len(gp_models))
    w = 0.19
    g_specs = [
        ("wb_clean", "Sade kod", C_TEAL, "", -1.5 * w),
        ("wb_decoy_long_refutable_guard", "Çürütülebilir kontrol", C_PURPLE, "//", -0.5 * w),
        ("wb_inv_code", "Zor koruyuculu kod", C_VERM, "", +0.5 * w),
        ("wb_inv_relu", "Opak ağırlık", C_BLUE, "", +1.5 * w),
    ]
    for vkey, lbl, col, hatch, xoff in g_specs:
        vals = np.array([get_exp2_cell(cells, mkey)["variants"][vkey]["acc"] for _, mkey in gp_models])
        ax1.bar(xg + xoff, vals - 0.5, bottom=0.5, width=w * 0.90, color=col, edgecolor=C_DARK, linewidth=0.45, hatch=hatch, label=lbl, zorder=3)

    bb_vals = [get_exp2_cell(cells, mkey)["bb_acc"] for _, mkey in gp_models]
    ax1.plot(xg, bb_vals, color=C_DARK, marker="D", linestyle="none", markersize=4.6,
             markerfacecolor="white", markeredgewidth=1.0, label="Kara-kutu (BB)", zorder=5)

    ax1.axhline(0.5, color=C_DARK, linewidth=0.7, zorder=2)
    ax1.axvline(1.5, color=C_GRAY, linestyle=":", linewidth=0.9, zorder=2)
    ax1.text(0.5, 1.035, "Gemma 4", ha="center", va="bottom", fontsize=6.2, color=C_DARK, fontweight="bold")
    ax1.text(3.5, 1.035, "Google Gemini", ha="center", va="bottom", fontsize=6.2, color=C_DARK, fontweight="bold")

    ax1.set_xticks(xg)
    ax1.set_xticklabels([m[0].replace("\n", " ") for m in gp_models], fontsize=5.8, rotation=32, ha="right", rotation_mode="anchor")
    ax1.set_ylim(0.47, 1.10)
    ax1.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax1.yaxis.set_major_formatter(tr_axis_fmt(1, signed=False))
    ax1.set_ylabel("Test doğruluğu (Acc)")
    ax1.set_title("(a) Gemini ve Gemma 4 Modellerinde Doğruluk", loc="left", fontsize=7.0, fontweight="bold", pad=7)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.50, -0.24), ncol=3, frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=5.7, columnspacing=0.9)

    all_15 = [
        ("Sonnet 4.5", "claude-sonnet-4-5-20250929"),
        ("Haiku 4.5", "claude-haiku-4-5-20251001"),
        ("Opus 4.6", "claude-opus-4-6"),
        ("Sonnet 4.6", "claude-sonnet-4-6"),
        ("Opus 4.8", "claude-opus-4-8"),
        ("Haiku 5.5", "claude-haiku-5-5"),
        ("Sonnet 5.5", "claude-sonnet-5-5"),
        ("Opus 5.5", "claude-opus-5-5"),
        ("Fable 5.2", "claude-fable-5-2"),
        ("Gemini 3.1 Flash-Lite", "gemini-3.1-flash-lite"),
        ("Gemini 3.5 Flash-Lite", "gemini-3.5-flash-lite"),
        ("Gemini 3.8 Flash", "gemini-3.8-flash"),
        ("Gemini 4 Carbon", "gemini-4-carbon"),
        ("Gemma 4 26B", "gemma-4-26b-a4b-it"),
        ("Gemma 4 31B", "gemma-4-31b-it"),
    ]
    fu_trap = {r["model"]: 100.0 * r["trap_refut"] for r in fu.get("part_a_refutable_extension", {}).get("main_condition_per_model", [])}
    y15 = np.arange(len(all_15), dtype=float)
    hb = 0.36
    code_dec, ref_dec = [], []
    for _, m in all_15:
        c = get_exp2_cell(cells, m)
        code_dec.append(100.0 * c["variants"]["wb_inv_code"]["decoy_rate"])
        if "wb_decoy_long_refutable_guard" in c["variants"]:
            ref_dec.append(100.0 * c["variants"]["wb_decoy_long_refutable_guard"]["decoy_rate"])
        else:
            ref_dec.append(fu_trap.get(m))

    ax2.barh(y15 - hb / 2, code_dec, height=hb, color=C_VERM, edgecolor=C_DARK, linewidth=0.45, label="Zor koruyuculu kod", zorder=3)
    ry = [yy + hb / 2 for yy, v in zip(y15, ref_dec) if v is not None]
    rv = [v for v in ref_dec if v is not None]
    ax2.barh(ry, rv, height=hb, color=C_PURPLE, edgecolor=C_DARK, linewidth=0.45, hatch="//", label="Çürütülebilir kontrol", zorder=3)
    for yi, val in zip(y15, code_dec):
        ax2.text(val + 1.6, yi - hb / 2, f"%{val:.0f}", va="center", ha="left", fontsize=5.3, color=C_DARK)
    for yi, val in zip(y15, ref_dec):
        if val is None:
            ax2.text(3.0, yi + hb / 2 + 0.02, "sınanamadı", va="center", ha="left", fontsize=5.1, color=C_GRAY, style="italic")
        else:
            ax2.text(val + 1.6, yi + hb / 2, f"%{val:.0f}", va="center", ha="left", fontsize=5.3, color=C_DARK)

    for ysep in (8.5, 12.5):
        ax2.axhline(ysep, color=C_DARK, linestyle="-", linewidth=0.6)
    for ylab, txt in ((8.0, "Anthropic"), (12.0, "Google Gemini"), (13.0, "Gemma 4")):
        ax2.text(124, ylab, txt, ha="right", va="center", fontsize=6.0, fontweight="bold", color=C_DARK)
    ax2.set_yticks(y15)
    ax2.set_yticklabels([r[0] for r in all_15], fontsize=5.9)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 126)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xlabel("Tuzak oranı (%)")
    ax2.set_title("(b) 15 Modelde Tuzak Oranı", loc="left", fontsize=7.0, fontweight="bold", pad=7)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.42, -0.24), ncol=2, frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=5.7)

    save_both(fig, "fig5_cross_provider_gemma")


# ==============================================================================
# ŞEKİL (fig6): Ek koşular — üç protokolde tuzak oranı ve doğruluk
# ==============================================================================
def make_fig6_protocols(fu: Dict[str, Any]) -> None:
    summ = fu.get("part_b_protocols", {}).get("summary_by_model_protocol", [])
    if not summ:
        return
    by = {(r["model"], r["protocol"]): r for r in summ}
    models = [
        ("Haiku 5.5", "claude-haiku-5-5"),
        ("Sonnet 5.5", "claude-sonnet-5-5"),
        ("Gemini 3.5\nFlash-Lite", "gemini-3.5-flash-lite"),
        ("Gemini 3.8\nFlash", "gemini-3.8-flash"),
    ]
    protos = [
        ("standard", "Standart", C_GRAY, ""),
        ("verify_first", "Sınama yönergesi", C_BLUE, "//"),
        ("harness_check", "Yanıt denetimi", C_AMBER, ".."),
    ]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(W_DOUBLE, 3.05),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.30},
    )
    x = np.arange(len(models))
    w = 0.25
    for j, (pkey, plbl, col, hatch) in enumerate(protos):
        xo = x + (j - 1) * w
        rows = [by[(m, pkey)] for _, m in models]
        # (a) trap rate
        tr = np.array([100.0 * r["trap_rate"] for r in rows])
        lo = np.array([100.0 * r["trap_ci95"][0] for r in rows])
        hi = np.array([100.0 * r["trap_ci95"][1] for r in rows])
        ax1.bar(xo, tr, width=w * 0.90, color=col, edgecolor=C_DARK, linewidth=0.5, hatch=hatch, label=plbl, zorder=3)
        ax1.errorbar(xo, tr, yerr=[tr - lo, hi - tr], fmt="none", ecolor=C_DARK, elinewidth=0.7, capsize=1.8, zorder=4)
        for xi, v, h in zip(xo, tr, hi):
            ax1.text(xi, h + 1.5, f"%{v:.0f}", ha="center", va="bottom", fontsize=5.6, color=C_DARK)
        # (b) accuracy on decoy code, bars from the 0.5 chance level
        ac = np.array([r["acc_decoy_code"] for r in rows])
        alo = np.array([r["acc_decoy_code_ci95"][0] for r in rows])
        ahi = np.array([r["acc_decoy_code_ci95"][1] for r in rows])
        ax2.bar(xo, ac - 0.5, bottom=0.5, width=w * 0.90, color=col, edgecolor=C_DARK, linewidth=0.5, hatch=hatch, zorder=3)
        ax2.errorbar(xo, ac, yerr=[np.maximum(0, ac - alo), np.maximum(0, ahi - ac)], fmt="none", ecolor=C_DARK, elinewidth=0.7, capsize=1.8, zorder=4)
        bb = [r["acc_bb"] for r in rows]
        ax2.plot(xo, bb, linestyle="none", marker="D", markersize=4.4, markerfacecolor="white", markeredgecolor=C_DARK,
                 markeredgewidth=1.0, label="Kara-kutu, aynı protokol" if j == 0 else None, zorder=6)

    ax1.set_xticks(x)
    ax1.set_xticklabels([m[0] for m in models])
    ax1.set_ylim(0, 72)
    ax1.set_ylabel("Tuzak oranı (%)")
    ax1.set_title("(a) Yanıltıcı Dallı Kodda Tuzak Oranı", loc="left", fontsize=7.2, fontweight="bold", pad=7)

    ax2.axhline(0.5, color=C_DARK, linewidth=0.7, zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[0] for m in models])
    ax2.set_ylim(0.47, 1.06)
    ax2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax2.yaxis.set_major_formatter(tr_axis_fmt(1, signed=False))
    ax2.set_ylabel("Test doğruluğu (Acc)")
    ax2.set_title("(b) Yanıltıcı Dallı Kodda Doğruluk", loc="left", fontsize=7.2, fontweight="bold", pad=7)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=4, frameon=True,
               facecolor="white", edgecolor="#D1D5DB", fontsize=6.0)

    save_both(fig, "fig6_protocols")


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    c3 = json.loads(C3_REPORT.read_text(encoding="utf-8"))
    # Kilitli veri kümesinden sonra yürütülen ek koşuların raporu (yoksa şekiller yalnızca kilitli veriyi gösterir).
    fu = json.loads(FOLLOWUP_REPORT.read_text(encoding="utf-8")) if FOLLOWUP_REPORT.exists() else {}
    make_fig0_workflow()
    make_fig1(c3)
    make_fig2(c3, fu)
    make_fig3(c3)
    make_fig4(c3)
    make_fig5_cross_provider(c3, fu)
    make_fig6_protocols(fu)
    print(f"Generated Turkish figures (PDF + 600 DPI PNG) in {FIG_DIR}")


if __name__ == "__main__":
    main()
