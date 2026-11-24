"""Figures of the paper, rendered from the main and follow-up reports.

    python -m blackwell_inversion.analysis.figures --lang en --out-dir figures

Reads `reports/report_c3_unified.json` and, when present, `reports/report_followup.json`;
writes vector PDFs (Type 42 fonts) and 600 DPI PNGs sized for a 185 mm double column.
`--lang tr` renders the Turkish manuscript figures, `--lang en` the same figures with
English labels. Needs matplotlib (the `figures` extra).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from ..paths import FIGURES_DIR, FOLLOWUP_REPORT_JSON, MAIN_REPORT_JSON  # noqa: E402

W_DOUBLE = 7.28

# Okabe-Ito colorblind-safe palette
C_BLUE = "#0072B2"
C_VERM = "#D55E00"
C_TEAL = "#009E73"
C_AMBER = "#E69F00"
C_PURPLE = "#CC79A7"
C_GRAY = "#5C646C"
C_DARK = "#1F2933"
LEGEND_EDGE = "#D1D5DB"
INVERSION_SHADE = "#FDF2EE"

NET_WB_ADVANTAGE_TR = r"Net Beyaz-Kutu Avantajı ($\Delta_{\mathrm{WB}} = \mathrm{WB} - \mathrm{BB}$)"
NET_WB_ADVANTAGE_EN = r"Net White-Box Advantage ($\Delta_{\mathrm{WB}} = \mathrm{WB} - \mathrm{BB}$)"

LABELS: Dict[str, Dict[str, Any]] = {
    "tr": {
        "fig0_boxes": [
            ("Test Üretimi", [
                ("Deney 1 (Arka Kapı)", "4-XOR kod, MBA ve ReLU"),
                ("Deney 2 (Yanıltıcı Dal)", "5 beyaz-kutu varyantı"),
                ("Kontrol", "Sızıntı testi (η = 0,50)"),
            ]),
            ("Yalıtılmış Ortam", [
                ("Sorgu Arayüzü", "Ortak BB/WB (q ≤ 80)"),
                ("Python Süreci", "Dosya yalnızca WB kipinde"),
                ("Eşit Bütçe", "Kip başına 3, 5 veya 8 tur"),
            ]),
            ("Değerlendirme", [
                ("Kapalı Modeller", "Claude (9) ve Gemini (4)"),
                ("Açık Ağırlık", "Gemma 4 (26B ve 31B)"),
                ("Klasik Referans", "8 saldırı (Kikuchi-2, SA)"),
            ]),
            ("Analiz", [
                ("İstatistik", "Fisher, permütasyon, Holm"),
                ("Kayıp Ayrıştırması", "Tuzak, kesilme, diğer"),
                ("Bütünlük", "7.631 deneme, test paketi"),
            ]),
        ],
        "fig0_tool_arrow": "Araç",
        "solvable_clean": "Çözülebilir: sade kod",
        "solvable_dense": "Çözülebilir: yoğun ağırlık",
        "hard_clean": "Zor: sade kod",
        "hard_dense": "Zor: yoğun ağırlık",
        "adv_axis_full": "Ayırt Etme Avantajı (Adv = TPR − FPR)",
        "adv_axis": "Ayırt Etme Avantajı (Adv)",
        "fig1a_title": "(a) Sertleştirilmiş 4-XOR Ailesi (N = 22–60)",
        "fig1b_regimes": ["m = 900 denklem\n(zor)", "m = 4.400 denklem\n(çözülebilir)"],
        "classical_attack": "Klasik saldırı (Canon + Kikuchi-2)",
        "haiku_clean": "Haiku 5.5 (sade kod)",
        "sonnet_clean": "Sonnet 5.5 (sade kod)",
        "haiku_sparse": "Haiku 5.5 (seyrek ağırlık)",
        "sonnet_sparse": "Sonnet 5.5 (seyrek ağırlık)",
        "fig1b_title": "(b) Sabit Boyutlu 4-XOR: Denklem Sayısının Etkisi",
        "groups": ["Birinci grup", "İkinci grup", "Üçüncü grup"],
        "clean_code": "Sade kod",
        "short_control": "Kısa kontrol",
        "refutable_control": "Çürütülebilir kontrol",
        "hard_guard_code": "Zor koruyuculu kod",
        "opaque_weights": "Opak ağırlık",
        "opus46_8k": "Opus 4.6, zor koruyuculu kod, token tavanı 8.192",
        "net_wb_advantage": NET_WB_ADVANTAGE_TR,
        "inversion": r"Ters etki ($\Delta_{\mathrm{WB}} < 0$)",
        "effort_ticks": ["Düşük\n(low)", "Orta\n(medium)", "Yüksek\n(high)"],
        "fig3a_ylabel": "Zor Koruyuculu Kodda Tuzak Oranı (%)",
        "fig3a_title": "(a) Efor Taraması: İki Sağlayıcıdan Dört Model",
        "turn_ticks": ["3 tur", "5 tur", "8 tur"],
        "haiku_hard_guard": "Haiku 5.5 · zor koruyuculu kod",
        "haiku_opaque": "Haiku 5.5 · opak ağırlık",
        "sonnet_hard_guard": "Sonnet 5.5 · zor koruyuculu kod",
        "sonnet_opaque": "Sonnet 5.5 · opak ağırlık",
        "fig3b_note": "8 turda iki modelde de\n$\\Delta_{\\mathrm{WB}} \\geq 0$ ve tuzak oranı %0",
        "fig3b_xlabel": "Kip başına konuşma turu sayısı",
        "fig3b_title": "(b) Tur Bütçesinin Etkisi (BB ve WB Eşit Turda)",
        "fig4a_rows": [
            "Haiku 5.5 (3 tur)",
            "Haiku 5.5 (high efor)",
            "Sonnet 5.5",
            "Sonnet 5.5 (Karar Listesi-3)",
            "Opus 4.6",
            "Gemini 3.8 Flash",
            "Gemma 4 31B",
        ],
        "fig4a_xlabel": "Beyaz-kutu test doğruluğu (Acc)",
        "fig4a_title": "(a) Dört Beyaz-Kutu Varyantının Doğruluğu",
        "fig4b_rows": [
            "Sonnet 4.5",
            "Opus 4.6 (tavan 4.096)",
            "Opus 4.6 (tavan 8.192)",
            "Haiku 5.5 (high efor)",
            "Sonnet 5.5",
            "Gemini 3.5 Flash-Lite\n(high efor)",
            "Gemma 4 31B",
            "Gemma 4 26B",
        ],
        "loss_decoy": "Tuzak kural seçildi",
        "loss_trunc": "Token kesilmesi",
        "loss_other": "Diğer yanlış yanıt veya geçersiz biçim",
        "fig4b_xlabel": "Doğruluk kaybı (1 − Acc)",
        "fig4b_title": "(b) Zor Koruyuculu Kodda Doğruluk Kaybının Bileşenleri",
        "black_box": "Kara-kutu (BB)",
        "test_accuracy": "Test doğruluğu (Acc)",
        "fig5a_title": "(a) Gemini ve Gemma 4 Modellerinde Doğruluk",
        "not_tested": "sınanamadı",
        "trap_rate": "Tuzak oranı (%)",
        "fig5b_title": "(b) 15 Modelde Tuzak Oranı",
        "protocols": ["Standart", "Sınama yönergesi", "Yanıt denetimi"],
        "bb_same_protocol": "Kara-kutu, aynı protokol",
        "fig6a_title": "(a) Yanıltıcı Dallı Kodda Tuzak Oranı",
        "fig6b_title": "(b) Yanıltıcı Dallı Kodda Doğruluk",
    },
    "en": {
        "fig0_boxes": [
            ("Test Generation", [
                ("Exp. 1 (Backdoor)", "4-XOR code, MBA and ReLU"),
                ("Exp. 2 (Decoy Branch)", "5 white-box variants"),
                ("Control", "Leak test (η = 0.50)"),
            ]),
            ("Sandbox", [
                ("Query Oracle", "Shared by BB/WB (q ≤ 80)"),
                ("Python Process", "Artifact only in WB mode"),
                ("Equal Budget", "3, 5 or 8 turns per mode"),
            ]),
            ("Evaluation", [
                ("Closed Models", "Claude (9) and Gemini (4)"),
                ("Open Weights", "Gemma 4 (26B and 31B)"),
                ("Classical Baselines", "8 attacks (Kikuchi-2, SA)"),
            ]),
            ("Analysis", [
                ("Statistics", "Fisher, permutation, Holm"),
                ("Loss Decomposition", "Decoy, truncation, other"),
                ("Integrity", "7,631 trials, test suite"),
            ]),
        ],
        "fig0_tool_arrow": "Tools",
        "solvable_clean": "Solvable: clean code",
        "solvable_dense": "Solvable: dense weights",
        "hard_clean": "Hard: clean code",
        "hard_dense": "Hard: dense weights",
        "adv_axis_full": "Distinguishing Advantage (Adv = TPR − FPR)",
        "adv_axis": "Distinguishing Advantage (Adv)",
        "fig1a_title": "(a) Hardened 4-XOR Family (N = 22–60)",
        "fig1b_regimes": ["m = 900 equations\n(hard)", "m = 4,400 equations\n(solvable)"],
        "classical_attack": "Classical attack (Canon + Kikuchi-2)",
        "haiku_clean": "Haiku 5.5 (clean code)",
        "sonnet_clean": "Sonnet 5.5 (clean code)",
        "haiku_sparse": "Haiku 5.5 (sparse weights)",
        "sonnet_sparse": "Sonnet 5.5 (sparse weights)",
        "fig1b_title": "(b) Fixed-Size 4-XOR: Effect of the Equation Count",
        "groups": ["First group", "Second group", "Third group"],
        "clean_code": "Clean code",
        "short_control": "Short control",
        "refutable_control": "Refutable control",
        "hard_guard_code": "Hard-guard code",
        "opaque_weights": "Opaque weights",
        "opus46_8k": "Opus 4.6, hard-guard code, token cap 8,192",
        "net_wb_advantage": NET_WB_ADVANTAGE_EN,
        "inversion": r"Inverse effect ($\Delta_{\mathrm{WB}} < 0$)",
        "effort_ticks": ["Low", "Medium", "High"],
        "fig3a_ylabel": "Trap Rate on Hard-Guard Code (%)",
        "fig3a_title": "(a) Effort Sweep: Four Models, Two Providers",
        "turn_ticks": ["3 turns", "5 turns", "8 turns"],
        "haiku_hard_guard": "Haiku 5.5 · hard-guard code",
        "haiku_opaque": "Haiku 5.5 · opaque weights",
        "sonnet_hard_guard": "Sonnet 5.5 · hard-guard code",
        "sonnet_opaque": "Sonnet 5.5 · opaque weights",
        "fig3b_note": "At 8 turns both models have\n$\\Delta_{\\mathrm{WB}} \\geq 0$ and a 0% trap rate",
        "fig3b_xlabel": "Conversation turns per mode",
        "fig3b_title": "(b) Effect of the Turn Budget (Equal for BB and WB)",
        "fig4a_rows": [
            "Haiku 5.5 (3 turns)",
            "Haiku 5.5 (high effort)",
            "Sonnet 5.5",
            "Sonnet 5.5 (Decision List-3)",
            "Opus 4.6",
            "Gemini 3.8 Flash",
            "Gemma 4 31B",
        ],
        "fig4a_xlabel": "White-box test accuracy (Acc)",
        "fig4a_title": "(a) Accuracy of Four White-Box Variants",
        "fig4b_rows": [
            "Sonnet 4.5",
            "Opus 4.6 (cap 4,096)",
            "Opus 4.6 (cap 8,192)",
            "Haiku 5.5 (high effort)",
            "Sonnet 5.5",
            "Gemini 3.5 Flash-Lite\n(high effort)",
            "Gemma 4 31B",
            "Gemma 4 26B",
        ],
        "loss_decoy": "Decoy rule chosen",
        "loss_trunc": "Token truncation",
        "loss_other": "Other wrong answer or invalid format",
        "fig4b_xlabel": "Accuracy loss (1 − Acc)",
        "fig4b_title": "(b) Accuracy Loss on Hard-Guard Code by Cause",
        "black_box": "Black box (BB)",
        "test_accuracy": "Test accuracy (Acc)",
        "fig5a_title": "(a) Accuracy of Gemini and Gemma 4 Models",
        "not_tested": "not tested",
        "trap_rate": "Trap rate (%)",
        "fig5b_title": "(b) Trap Rate across 15 Models",
        "protocols": ["Standard", "Verify-first instruction", "Answer check"],
        "bb_same_protocol": "Black box, same protocol",
        "fig6a_title": "(a) Trap Rate on Code with a Decoy Branch",
        "fig6b_title": "(b) Accuracy on Code with a Decoy Branch",
    },
}


@dataclass(frozen=True)
class Locale:
    """Labels and number formatting for one figure language."""

    labels: Dict[str, Any]
    decimal_comma: bool
    percent_prefix: bool

    def __getitem__(self, key: str) -> Any:
        return self.labels[key]

    def num(self, val: float, decimals: int = 3, signed: bool = False) -> str:
        """Number with a typographic minus, and a decimal comma where the language uses one."""
        if abs(val) < 1e-12:
            val = 0.0
        txt = f"{val:+.{decimals}f}" if (signed and val != 0.0) else f"{val:.{decimals}f}"
        txt = txt.replace("-", "−")
        return txt.replace(".", ",") if self.decimal_comma else txt

    def axis_fmt(self, decimals: int = 1, signed: bool = False) -> FuncFormatter:
        return FuncFormatter(lambda x, _pos: self.num(x, decimals=decimals, signed=signed))

    def pct(self, val: float) -> str:
        """Whole-number percentage ("%12" in Turkish, "12%" in English)."""
        return f"%{val:.0f}" if self.percent_prefix else f"{val:.0f}%"


LOCALES = {
    "tr": Locale(LABELS["tr"], decimal_comma=True, percent_prefix=True),
    "en": Locale(LABELS["en"], decimal_comma=False, percent_prefix=False),
}


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


def save_both(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}")
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
    """The Exp 2 cell of the main report; a medium-effort lookup falls back to effort 'default'."""
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


def _legend_below(ax: Any, **kwargs: Any) -> None:
    ax.legend(loc="upper center", frameon=True, facecolor="white", edgecolor=LEGEND_EDGE, **kwargs)


# ==============================================================================
# fig0: workflow of the study
# ==============================================================================
def make_fig0_workflow(loc: Locale, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.25))
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 3.10)
    ax.axis("off")

    # Single monochrome slate-navy theme
    PRIMARY = "#1E3A5F"
    TINT = "#F4F7FA"
    BORDER = "#94A3B8"
    INK = "#1E293B"
    MUTED = "#475569"

    text_bounds: List[tuple] = []
    geometry = [(0.06, 2.12, False), (2.60, 2.12, True), (5.26, 2.12, False), (7.80, 2.14, False)]
    boxes = [
        {"x": x, "w": w, "dashed": dashed, "num": str(i + 1), "title": title, "items": items}
        for i, ((x, w, dashed), (title, items)) in enumerate(zip(geometry, loc["fig0_boxes"]))
    ]

    for b in boxes:
        x, w = b["x"], b["w"]
        y, h = 0.20, 2.70

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

        hdr_h = 0.56
        hdr = FancyBboxPatch(
            (x, y + h - hdr_h), w, hdr_h,
            boxstyle="round,pad=0.015,rounding_size=0.09",
            facecolor=TINT, edgecolor="none", zorder=3,
        )
        ax.add_patch(hdr)
        ax.plot([x + 0.04, x + w - 0.04], [y + h - hdr_h, y + h - hdr_h],
                color=BORDER, linewidth=0.6, zorder=4)

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

        row_ys = [1.84, 1.18, 0.54]
        for (head, desc), ry in zip(b["items"], row_ys):
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
    draw_conn(4.76, 5.22, style="<|-|>", label=loc["fig0_tool_arrow"])
    draw_conn(7.42, 7.76)

    # Every text must stay inside its card.
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

    save_both(fig, out_dir, "fig0_workflow")


# ==============================================================================
# fig1: Exp 1, hardened 4-XOR family and the fixed-size m phase transition
# ==============================================================================
def make_fig1(loc: Locale, out_dir: Path, c3: Dict[str, Any]) -> None:
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
        (loc["solvable_clean"], "transparent", "code_clean", C_BLUE, ""),
        (loc["solvable_dense"], "transparent", "relu_weights_dense", C_TEAL, "//"),
        (loc["hard_clean"], "vbb_hard", "code_clean", C_VERM, ""),
        (loc["hard_dense"], "vbb_hard", "relu_weights_dense", C_AMBER, "xx"),
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
    ax1.set_ylabel(loc["adv_axis_full"])
    ax1.set_ylim(-0.52, 1.16)
    ax1.yaxis.set_major_formatter(loc.axis_fmt(1, signed=True))
    ax1.set_title(loc["fig1a_title"], loc="left", fontsize=7.5, fontweight="bold", pad=7)
    _legend_below(ax1, bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.1, handlelength=1.4)

    # Panel (b): only m varies (n=120, k=4, eta=0.20).
    regimes_b = loc["fig1b_regimes"]
    xb = np.arange(len(regimes_b))
    series_b = [
        (loc["classical_attack"], [-0.04, 1.00], [0.0, 0.0], [0.0, 0.0], C_GRAY, "s", "--"),
        (loc["haiku_clean"], [], [], [], C_VERM, "o", "-"),
        (loc["sonnet_clean"], [], [], [], C_BLUE, "D", "-"),
        (loc["haiku_sparse"], [], [], [], C_AMBER, "^", ":"),
        (loc["sonnet_sparse"], [], [], [], C_TEAL, "v", ":"),
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
    # Ticks every 0.2, as in panel (a), so the one-decimal labels are exact.
    ax2.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.yaxis.set_major_formatter(loc.axis_fmt(1, signed=True))
    ax2.set_ylabel(loc["adv_axis"])
    ax2.set_title(loc["fig1b_title"], loc="left", fontsize=7.4, fontweight="bold", pad=7)
    _legend_below(ax2, bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=5.9)

    save_both(fig, out_dir, "fig1_exp1_vbb_hierarchy")


# ==============================================================================
# fig2: Delta_WB by variant on the nine Anthropic models, in group order
# ==============================================================================
def make_fig2(loc: Locale, out_dir: Path, c3: Dict[str, Any], fu: Dict[str, Any]) -> None:
    cells = c3["exp2"]["cells"]
    # The three groups of the text; release order within each group.
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
    groups = [(glbl, i0, i1) for glbl, (i0, i1) in zip(loc["groups"], [(0, 1), (2, 6), (7, 8)])]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.25))

    ax.axhspan(-0.52, 0.0, color=INVERSION_SHADE, alpha=0.55, zorder=0)
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
        ("wb_clean", loc["clean_code"], C_TEAL, "^", 4.6, -0.28),
        ("wb_decoy_trivial_guard", loc["short_control"], C_GRAY, "s", 4.2, -0.14),
        ("wb_decoy_long_refutable_guard", loc["refutable_control"], C_PURPLE, "v", 4.8, 0.0),
        ("wb_inv_code", loc["hard_guard_code"], C_VERM, "o", 5.4, +0.14),
        ("wb_inv_relu", loc["opaque_weights"], C_BLUE, "D", 4.2, +0.28),
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
               label=loc["opus46_8k"], zorder=7)

    for (glbl, i0, i1) in groups:
        ax.text((i0 + i1) / 2.0, 0.575, glbl, ha="center", va="center", fontsize=6.6, fontweight="bold", color=C_DARK)
    for xsep in (1.5, 6.5):
        ax.axvline(xsep, color=C_DARK, linestyle="-", linewidth=0.7, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in models])
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylabel(loc["net_wb_advantage"])
    ax.set_ylim(-0.52, 0.62)
    ax.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    ax.yaxis.set_major_formatter(loc.axis_fmt(1, signed=True))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, facecolor="white", edgecolor=LEGEND_EDGE, framealpha=0.95, fontsize=6.0)
    ax.text(0.995, 0.03, loc["inversion"], transform=ax.transAxes,
            ha="right", fontsize=6.6, color=C_VERM, fontweight="bold")

    save_both(fig, out_dir, "fig2_exp2_chronology")


# ==============================================================================
# fig3: effort sweep on four models and the equal-turn budget phase
# ==============================================================================
def make_fig3(loc: Locale, out_dir: Path, c3: Dict[str, Any]) -> None:
    cells = c3["exp2"]["cells"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(W_DOUBLE, 3.45),
        gridspec_kw={"width_ratios": [1.04, 1.0], "wspace": 0.30},
    )

    efforts = ["low", "medium", "high"]
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
            # Hand-placed offsets keep the value labels of crossing lines apart.
            if mkey == "claude-opus-4-8" and xi == 0:
                dy = -5.5
            elif mkey == "gemini-3.5-flash-lite" and xi == 1:
                dy = -6.0
            elif mkey == "claude-opus-4-8" and xi == 1:
                dy = +3.2
            else:
                dy = +3.0
            ax1.text(xi, r + dy, loc.pct(r), ha="center", va="bottom" if dy > 0 else "top",
                     fontsize=6.3, fontweight="bold", color=col)

    ax1.set_xticks(xe)
    ax1.set_xticklabels(loc["effort_ticks"])
    ax1.set_xlim(-0.32, 2.32)
    ax1.set_ylim(-2, 78)
    ax1.set_ylabel(loc["fig3a_ylabel"])
    ax1.set_title(loc["fig3a_title"], loc="left", fontsize=7.2, fontweight="bold", pad=7)
    _legend_below(ax1, bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=5.8)

    tracks_b = list(zip(loc["turn_ticks"], ["matched_total_budget", "matched_medium_budget", "matched_wide_equal_budget"]))
    xb = np.arange(len(tracks_b))
    ax2.axhline(0.0, color=C_DARK, linewidth=0.8, zorder=1)
    ax2.axhspan(-0.40, 0.0, color=INVERSION_SHADE, alpha=0.5, zorder=0)

    b_series = [
        (loc["haiku_hard_guard"], "claude-haiku-5-5", "wb_inv_code", C_VERM, "o", "-", 1.7, -0.05),
        (loc["haiku_opaque"], "claude-haiku-5-5", "wb_inv_relu", C_AMBER, "s", "--", 1.35, -0.02),
        (loc["sonnet_hard_guard"], "claude-sonnet-5-5", "wb_inv_code", C_BLUE, "D", "-", 1.7, +0.02),
        (loc["sonnet_opaque"], "claude-sonnet-5-5", "wb_inv_relu", C_TEAL, "^", ":", 1.35, +0.05),
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
        loc["fig3b_note"],
        xy=(2.0, 0.0), xytext=(0.95, -0.33),
        fontsize=5.9, color=C_DARK, fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": C_DARK, "lw": 0.75},
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#ECFDF5", "edgecolor": C_TEAL, "lw": 0.75},
        zorder=6,
    )
    ax2.set_xticks(xb)
    ax2.set_xticklabels([t[0] for t in tracks_b])
    ax2.set_xlabel(loc["fig3b_xlabel"])
    ax2.set_xlim(-0.32, 2.32)
    ax2.set_ylim(-0.40, 0.28)
    ax2.set_ylabel(loc["net_wb_advantage"])
    ax2.yaxis.set_major_formatter(loc.axis_fmt(1, signed=True))
    ax2.set_title(loc["fig3b_title"], loc="left", fontsize=7.2, fontweight="bold", pad=7)
    _legend_below(ax2, bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=5.8)

    save_both(fig, out_dir, "fig3_exp2_effort_budget")


# ==============================================================================
# fig4: causal controls (B8) and the three-way loss decomposition
# ==============================================================================
def make_fig4(loc: Locale, out_dir: Path, c3: Dict[str, Any]) -> None:
    cells = c3["exp2"]["cells"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(W_DOUBLE, 3.85),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.56},
    )

    cells_a = [
        ("claude-haiku-5-5", "medium", "matched_total_budget", "majority3"),
        ("claude-haiku-5-5", "high", "matched_medium_budget", "majority3"),
        ("claude-sonnet-5-5", "medium", "matched_medium_budget", "majority3"),
        ("claude-sonnet-5-5", "medium", "matched_medium_budget", "dlist3"),
        ("claude-opus-4-6", "medium", "matched_medium_budget", "majority3"),
        ("gemini-3.8-flash", "medium", "matched_medium_budget", "majority3"),
        ("gemma-4-31b-it", "medium", "matched_medium_budget", "majority3"),
    ]
    rows_a = [(lbl, *cell) for lbl, cell in zip(loc["fig4a_rows"], cells_a)]
    y = np.arange(len(rows_a))
    h = 0.19
    v_specs = [
        ("wb_inv_relu", loc["opaque_weights"], C_BLUE, "", +1.5 * h),
        ("wb_decoy_trivial_guard", loc["short_control"], C_GRAY, "", +0.5 * h),
        ("wb_decoy_long_refutable_guard", loc["refutable_control"], C_PURPLE, "//", -0.5 * h),
        ("wb_inv_code", loc["hard_guard_code"], C_VERM, "", -1.5 * h),
    ]
    for vkey, lbl, col, hatch, yoff in v_specs:
        accs = [get_exp2_cell(cells, m, effort=ef, track=trk, task=tsk)["variants"][vkey]["acc"] for _, m, ef, trk, tsk in rows_a]
        ax1.barh(y + yoff, np.array(accs) - 0.5, left=0.5, height=h * 0.90, color=col, edgecolor=C_DARK, linewidth=0.45, hatch=hatch, label=lbl, zorder=3)

    ax1.set_yticks(y)
    ax1.set_yticklabels([r[0] for r in rows_a], fontsize=6.6)
    ax1.invert_yaxis()
    ax1.set_xlim(0.5, 1.03)
    ax1.xaxis.set_major_formatter(loc.axis_fmt(1, signed=False))
    ax1.set_xlabel(loc["fig4a_xlabel"])
    ax1.set_title(loc["fig4a_title"], loc="left", fontsize=7.0, fontweight="bold", pad=7)
    _legend_below(ax1, bbox_to_anchor=(0.48, -0.17), ncol=2, fontsize=5.8)

    cells_b = [
        ("claude-sonnet-4-5-20250929", "medium", "4096"),
        ("claude-opus-4-6", "medium", "4096"),
        ("claude-opus-4-6", "medium", "8192"),
        ("claude-haiku-5-5", "high", "4096"),
        ("claude-sonnet-5-5", "medium", "4096"),
        ("gemini-3.5-flash-lite", "high", "4096"),
        ("gemma-4-31b-it", "medium", "4096"),
        ("gemma-4-26b-a4b-it", "medium", "4096"),
    ]
    rows_b = [(lbl, *cell) for lbl, cell in zip(loc["fig4b_rows"], cells_b)]
    yb = np.arange(len(rows_b))
    ld, lt, lo = [], [], []
    for _, m, ef, mt in rows_b:
        c = get_exp2_cell(cells, m, effort=ef, mt=mt)
        v = c["variants"]["wb_inv_code"]
        ld.append(v["loss_decoy"])
        lt.append(v["loss_trunc"])
        lo.append(v["loss_other"])
    ld_a, lt_a, lo_a = np.array(ld), np.array(lt), np.array(lo)

    ax2.barh(yb, ld_a, height=0.62, color=C_VERM, edgecolor=C_DARK, linewidth=0.5, label=loc["loss_decoy"])
    ax2.barh(yb, lt_a, left=ld_a, height=0.62, color=C_AMBER, edgecolor=C_DARK, linewidth=0.5, hatch="//", label=loc["loss_trunc"])
    ax2.barh(yb, lo_a, left=ld_a + lt_a, height=0.62, color=C_GRAY, edgecolor=C_DARK, linewidth=0.5, hatch="..", label=loc["loss_other"])

    for yi, tot in zip(yb, ld_a + lt_a + lo_a):
        ax2.text(tot + 0.012, yi, loc.num(tot, 2), va="center", ha="left", fontsize=6.2, color=C_DARK, fontweight="bold")

    ax2.set_yticks(yb)
    ax2.set_yticklabels([r[0] for r in rows_b], fontsize=6.6)
    ax2.invert_yaxis()
    ax2.set_xlim(0.0, 0.58)
    ax2.xaxis.set_major_formatter(loc.axis_fmt(1, signed=False))
    ax2.set_xlabel(loc["fig4b_xlabel"])
    ax2.set_title(loc["fig4b_title"], loc="left", fontsize=7.0, fontweight="bold", pad=7)
    _legend_below(ax2, bbox_to_anchor=(0.48, -0.17), ncol=2, fontsize=5.8)

    save_both(fig, out_dir, "fig4_transparency_illusion")


# ==============================================================================
# fig5: Gemini and Gemma 4 accuracy; trap rates on 15 models
# ==============================================================================
def make_fig5_cross_provider(loc: Locale, out_dir: Path, c3: Dict[str, Any], fu: Dict[str, Any]) -> None:
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
        ("wb_clean", loc["clean_code"], C_TEAL, "", -1.5 * w),
        ("wb_decoy_long_refutable_guard", loc["refutable_control"], C_PURPLE, "//", -0.5 * w),
        ("wb_inv_code", loc["hard_guard_code"], C_VERM, "", +0.5 * w),
        ("wb_inv_relu", loc["opaque_weights"], C_BLUE, "", +1.5 * w),
    ]
    for vkey, lbl, col, hatch, xoff in g_specs:
        vals = np.array([get_exp2_cell(cells, mkey)["variants"][vkey]["acc"] for _, mkey in gp_models])
        ax1.bar(xg + xoff, vals - 0.5, bottom=0.5, width=w * 0.90, color=col, edgecolor=C_DARK, linewidth=0.45, hatch=hatch, label=lbl, zorder=3)

    bb_vals = [get_exp2_cell(cells, mkey)["bb_acc"] for _, mkey in gp_models]
    ax1.plot(xg, bb_vals, color=C_DARK, marker="D", linestyle="none", markersize=4.6,
             markerfacecolor="white", markeredgewidth=1.0, label=loc["black_box"], zorder=5)

    ax1.axhline(0.5, color=C_DARK, linewidth=0.7, zorder=2)
    ax1.axvline(1.5, color=C_GRAY, linestyle=":", linewidth=0.9, zorder=2)
    ax1.text(0.5, 1.035, "Gemma 4", ha="center", va="bottom", fontsize=6.2, color=C_DARK, fontweight="bold")
    ax1.text(3.5, 1.035, "Google Gemini", ha="center", va="bottom", fontsize=6.2, color=C_DARK, fontweight="bold")

    ax1.set_xticks(xg)
    ax1.set_xticklabels([m[0].replace("\n", " ") for m in gp_models], fontsize=5.8, rotation=32, ha="right", rotation_mode="anchor")
    ax1.set_ylim(0.47, 1.10)
    ax1.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax1.yaxis.set_major_formatter(loc.axis_fmt(1, signed=False))
    ax1.set_ylabel(loc["test_accuracy"])
    ax1.set_title(loc["fig5a_title"], loc="left", fontsize=7.0, fontweight="bold", pad=7)
    _legend_below(ax1, bbox_to_anchor=(0.50, -0.24), ncol=3, fontsize=5.7, columnspacing=0.9)

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
    code_dec: List[float] = []
    ref_dec: List[Optional[float]] = []
    for _, m in all_15:
        c = get_exp2_cell(cells, m)
        code_dec.append(100.0 * c["variants"]["wb_inv_code"]["decoy_rate"])
        if "wb_decoy_long_refutable_guard" in c["variants"]:
            ref_dec.append(100.0 * c["variants"]["wb_decoy_long_refutable_guard"]["decoy_rate"])
        else:
            ref_dec.append(fu_trap.get(m))

    ax2.barh(y15 - hb / 2, code_dec, height=hb, color=C_VERM, edgecolor=C_DARK, linewidth=0.45, label=loc["hard_guard_code"], zorder=3)
    ry = [yy + hb / 2 for yy, v in zip(y15, ref_dec) if v is not None]
    rv = [v for v in ref_dec if v is not None]
    ax2.barh(ry, rv, height=hb, color=C_PURPLE, edgecolor=C_DARK, linewidth=0.45, hatch="//", label=loc["refutable_control"], zorder=3)
    for yi, val in zip(y15, code_dec):
        ax2.text(val + 1.6, yi - hb / 2, loc.pct(val), va="center", ha="left", fontsize=5.3, color=C_DARK)
    for yi, val in zip(y15, ref_dec):
        if val is None:
            ax2.text(3.0, yi + hb / 2 + 0.02, loc["not_tested"], va="center", ha="left", fontsize=5.1, color=C_GRAY, style="italic")
        else:
            ax2.text(val + 1.6, yi + hb / 2, loc.pct(val), va="center", ha="left", fontsize=5.3, color=C_DARK)

    for ysep in (8.5, 12.5):
        ax2.axhline(ysep, color=C_DARK, linestyle="-", linewidth=0.6)
    for ylab, txt in ((8.0, "Anthropic"), (12.0, "Google Gemini"), (13.0, "Gemma 4")):
        ax2.text(124, ylab, txt, ha="right", va="center", fontsize=6.0, fontweight="bold", color=C_DARK)
    ax2.set_yticks(y15)
    ax2.set_yticklabels([r[0] for r in all_15], fontsize=5.9)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 126)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xlabel(loc["trap_rate"])
    ax2.set_title(loc["fig5b_title"], loc="left", fontsize=7.0, fontweight="bold", pad=7)
    _legend_below(ax2, bbox_to_anchor=(0.42, -0.24), ncol=2, fontsize=5.7)

    save_both(fig, out_dir, "fig5_cross_provider_gemma")


# ==============================================================================
# fig6: follow-up runs, trap rate and accuracy under three protocols
# ==============================================================================
def make_fig6_protocols(loc: Locale, out_dir: Path, fu: Dict[str, Any]) -> None:
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
        (pkey, plbl, col, hatch)
        for (pkey, col, hatch), plbl in zip(
            [("standard", C_GRAY, ""), ("verify_first", C_BLUE, "//"), ("harness_check", C_AMBER, "..")],
            loc["protocols"],
        )
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
            ax1.text(xi, h + 1.5, loc.pct(v), ha="center", va="bottom", fontsize=5.6, color=C_DARK)
        # (b) accuracy on decoy code, bars from the 0.5 chance level
        ac = np.array([r["acc_decoy_code"] for r in rows])
        alo = np.array([r["acc_decoy_code_ci95"][0] for r in rows])
        ahi = np.array([r["acc_decoy_code_ci95"][1] for r in rows])
        ax2.bar(xo, ac - 0.5, bottom=0.5, width=w * 0.90, color=col, edgecolor=C_DARK, linewidth=0.5, hatch=hatch, zorder=3)
        ax2.errorbar(xo, ac, yerr=[np.maximum(0, ac - alo), np.maximum(0, ahi - ac)], fmt="none", ecolor=C_DARK, elinewidth=0.7, capsize=1.8, zorder=4)
        bb = [r["acc_bb"] for r in rows]
        ax2.plot(xo, bb, linestyle="none", marker="D", markersize=4.4, markerfacecolor="white", markeredgecolor=C_DARK,
                 markeredgewidth=1.0, label=loc["bb_same_protocol"] if j == 0 else None, zorder=6)

    ax1.set_xticks(x)
    ax1.set_xticklabels([m[0] for m in models])
    ax1.set_ylim(0, 72)
    ax1.set_ylabel(loc["trap_rate"])
    ax1.set_title(loc["fig6a_title"], loc="left", fontsize=7.2, fontweight="bold", pad=7)

    ax2.axhline(0.5, color=C_DARK, linewidth=0.7, zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[0] for m in models])
    ax2.set_ylim(0.47, 1.06)
    ax2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax2.yaxis.set_major_formatter(loc.axis_fmt(1, signed=False))
    ax2.set_ylabel(loc["test_accuracy"])
    ax2.set_title(loc["fig6b_title"], loc="left", fontsize=7.2, fontweight="bold", pad=7)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=4, frameon=True,
               facecolor="white", edgecolor=LEGEND_EDGE, fontsize=6.0)

    save_both(fig, out_dir, "fig6_protocols")


def render_all(lang: str, out_dir: Path, main_report: Path, followup_report: Path) -> None:
    """Render every figure in `lang` into `out_dir`."""
    loc = LOCALES[lang]
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    c3 = json.loads(main_report.read_text(encoding="utf-8"))
    # Without the follow-up report the figures show the main data only.
    fu = json.loads(followup_report.read_text(encoding="utf-8")) if followup_report.exists() else {}
    make_fig0_workflow(loc, out_dir)
    make_fig1(loc, out_dir, c3)
    make_fig2(loc, out_dir, c3, fu)
    make_fig3(loc, out_dir, c3)
    make_fig4(loc, out_dir, c3)
    make_fig5_cross_provider(loc, out_dir, c3, fu)
    make_fig6_protocols(loc, out_dir, fu)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Render the paper figures (PDF and 600 DPI PNG) from the reports.")
    ap.add_argument("--lang", choices=sorted(LOCALES), default="en", help="Label language.")
    ap.add_argument("--out-dir", default=str(FIGURES_DIR), help="Output directory.")
    ap.add_argument("--main-report", default=str(MAIN_REPORT_JSON), help="report_c3_unified.json path.")
    ap.add_argument("--followup-report", default=str(FOLLOWUP_REPORT_JSON), help="report_followup.json path.")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    render_all(args.lang, out_dir, Path(args.main_report), Path(args.followup_report))
    print(f"Generated figures ({args.lang}, PDF + 600 DPI PNG) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
