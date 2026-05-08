# ============================================================
# volcano_plot.py
# Reusable module — Volcano Plot for DEG Analysis
#
# Public API:
#   build_volcano_plot(df, fc_threshold, padj_threshold,
#                      top_n_labels, output_path)
#
# Expected DataFrame columns:
#   gene            — HUGO gene symbol
#   log2FC          — log2 fold change (tumor / normal)
#   p_adj           — BH-corrected p-value
#   category        — 'Upregulated' | 'Downregulated' | 'Not Significant'
#   neg_log10_padj  — -log10(p_adj), pre-computed by caller
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os


def build_volcano_plot(df, fc_threshold, padj_threshold,
                       top_n_labels, output_path):
    """
    Draw and save a volcano plot from a pre-categorised gene DataFrame.

    Parameters
    ----------
    df              : pd.DataFrame  — full gene table with 'category' and
                                      'neg_log10_padj' columns already set
    fc_threshold    : float         — |log2FC| cutoff used for categorisation
    padj_threshold  : float         — p_adj cutoff used for categorisation
    top_n_labels    : int           — how many top DEGs to annotate by name
    output_path     : str           — file path to save the PNG figure
    """

    # ── Count categories ──────────────────────────────────────
    up   = df[df['category'] == 'Upregulated']
    down = df[df['category'] == 'Downregulated']
    ns   = df[df['category'] == 'Not Significant']

    n_up, n_down, n_ns = len(up), len(down), len(ns)

    # Top DEGs to label: highest -log10(p_adj) across both directions
    all_degs = df[df['category'] != 'Not Significant']
    top_degs = all_degs.nlargest(top_n_labels, 'neg_log10_padj')

    # ── Figure setup ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    # Layer 1 — Not significant (gray, small, transparent)
    ax.scatter(
        ns['log2FC'], ns['neg_log10_padj'],
        color='#BBBBBB', alpha=0.35, s=7, linewidths=0,
        zorder=1, rasterized=True
    )

    # Layer 2 — Downregulated (blue)
    ax.scatter(
        down['log2FC'], down['neg_log10_padj'],
        color='#2166AC', alpha=0.75, s=22,
        linewidths=0.3, edgecolors='#1a4f88', zorder=2
    )

    # Layer 3 — Upregulated (red)
    ax.scatter(
        up['log2FC'], up['neg_log10_padj'],
        color='#D6604D', alpha=0.75, s=22,
        linewidths=0.3, edgecolors='#a8382a', zorder=3
    )

    # ── Threshold lines ───────────────────────────────────────
    sig_y = -np.log10(padj_threshold)

    ax.axhline(y=sig_y,          color='#444444', linestyle='--', linewidth=0.9, alpha=0.7, zorder=4)
    ax.axvline(x= fc_threshold,  color='#444444', linestyle='--', linewidth=0.9, alpha=0.7, zorder=4)
    ax.axvline(x=-fc_threshold,  color='#444444', linestyle='--', linewidth=0.9, alpha=0.7, zorder=4)

    # ── Threshold line annotations ────────────────────────────
    y_max = ax.get_ylim()[1]
    x_max = ax.get_xlim()[1]

    ax.text(x_max * 0.98, sig_y + 0.3,
            f'p_adj = {padj_threshold}',
            fontsize=7.5, color='#444444', ha='right', va='bottom', style='italic')

    for sign in (1, -1):
        ha = 'left' if sign == 1 else 'right'
        ax.text(sign * fc_threshold + sign * 0.08, y_max * 0.97,
                f'|FC| = {2**fc_threshold:.0f}×',
                fontsize=7.5, color='#444444', ha=ha, va='top', style='italic')

    # ── Gene labels for top DEGs ──────────────────────────────
    # Alternating offsets to reduce label overlap
    offsets = [
        ( 0.6,  1.2), (-0.6,  1.2), ( 1.0,  0.5), (-1.0,  0.5),
        ( 0.6, -1.0), (-0.6, -1.0), ( 1.2,  1.0), (-1.2,  1.0),
        ( 0.8,  1.8), (-0.8,  1.8), ( 0.3,  2.0), (-0.3,  2.0),
        ( 1.5,  0.3), (-1.5,  0.3), ( 0.0,  2.2),
    ]

    for idx, (_, row) in enumerate(top_degs.iterrows()):
        ox, oy = offsets[idx % len(offsets)]
        color  = '#D6604D' if row['category'] == 'Upregulated' else '#2166AC'
        ax.annotate(
            row['gene'],
            xy     =(row['log2FC'], row['neg_log10_padj']),
            xytext =(row['log2FC'] + ox, row['neg_log10_padj'] + oy),
            fontsize=6.8, fontweight='bold', color=color,
            ha='center', va='center', zorder=5,
            arrowprops=dict(arrowstyle='-', color='#888888', lw=0.6)
        )

    # ── Legend ────────────────────────────────────────────────
    patch_up   = mpatches.Patch(color='#D6604D', label=f'Upregulated ({n_up:,})')
    patch_down = mpatches.Patch(color='#2166AC', label=f'Downregulated ({n_down:,})')
    patch_ns   = mpatches.Patch(color='#BBBBBB', label=f'Not Significant ({n_ns:,})')
    line_thr   = mlines.Line2D(
        [], [], color='#444444', linestyle='--', linewidth=0.9,
        label=f'Thresholds: p_adj = {padj_threshold},  |log₂FC| = {fc_threshold}'
    )
    ax.legend(
        handles=[patch_up, patch_down, patch_ns, line_thr],
        loc='upper left', fontsize=9,
        framealpha=0.85, edgecolor='#CCCCCC'
    )

    # ── Stats text box ────────────────────────────────────────
    stats_text = (
        f"Total genes tested: {len(df):,}\n"
        f"DEGs (dual-threshold): {n_up + n_down:,}\n"
        f"  ↑ Upregulated:   {n_up:,}\n"
        f"  ↓ Downregulated: {n_down:,}"
    )
    ax.text(
        0.98, 0.98, stats_text,
        transform=ax.transAxes, fontsize=8.5,
        va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#CCCCCC', alpha=0.9)
    )

    # ── Axes formatting ───────────────────────────────────────
    ax.set_xlabel('log₂ Fold Change  (Tumor / Normal)', fontsize=12, labelpad=8)
    ax.set_ylabel('−log₁₀ (adjusted p-value)',          fontsize=12, labelpad=8)
    ax.set_title(
        'Volcano Plot — LUSC Tumor vs. Normal  (n = 51 paired samples)\n'
        f'DEG Criteria: |log₂FC| > {fc_threshold}  and  FDR-adjusted p < {padj_threshold}',
        fontsize=13, fontweight='bold', pad=14
    )
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5, color='#CCCCCC')
    ax.spines[['top', 'right']].set_visible(False)

    # ── Save ──────────────────────────────────────────────────
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"  Volcano plot saved: {output_path}")
    return top_degs
