# ============================================================
# fold_change_analysis.py
# Main entry point — Fold Change Analysis & Volcano Plot
#
# Pipeline:
#   1. Load Member 1's output (all_genes_results.csv)
#   2. Apply dual-threshold DEG filtering (|log2FC|>1, p_adj<0.05)
#   3. Compare filtered list against Member 1's p-value-only list
#   4. Draw and save the volcano plot  (via volcano_plot.py)
#   5. Export final DEG list for Member 3 / GSEA
#
# Run:
#   python fold_change_analysis.py
# ============================================================

import pandas as pd
import numpy as np
import os

from volcano_plot import build_volcano_plot

# ── Configuration ─────────────────────────────────────────────
FC_THRESHOLD   = 1.0    # |log2FC| > 1  ->  at least 2x fold change
PADJ_THRESHOLD = 0.05   # BH-corrected FDR threshold
TOP_N_LABELS   = 15     # gene names to annotate on the volcano plot

INPUT_CSV      = "outputs/all_genes_results.csv"
OUTPUT_DEG_CSV = "outputs/DEGs_volcano_filtered.csv"
OUTPUT_FIGURE  = "outputs/figures/volcano_plot.png"

# ═══════════════════════════════════════════════════════════════
# STEP 1 — Load & Validate Member 1's Output
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("FOLD CHANGE ANALYSIS & VOLCANO PLOT")
print("=" * 60)

assert os.path.exists(INPUT_CSV), (
    f"\n[ERROR] '{INPUT_CSV}' not found.\n"
    "Run main.py (Member 1's script) first to generate outputs/."
)

df = pd.read_csv(INPUT_CSV)

required_cols = {'gene', 'p_adj', 'log2FC', 'significant'}
missing = required_cols - set(df.columns)
assert not missing, f"[ERROR] Missing columns in input: {missing}"

for col in ['log2FC', 'p_adj']:
    assert df[col].isna().sum() == 0, \
        f"[ERROR] Column '{col}' has NaN values — check Member 1's output."

print(f"\n  Input file:   {INPUT_CSV}")
print(f"  Total genes:  {len(df):,}")
print()

# ═══════════════════════════════════════════════════════════════
# STEP 2 — Dual-Threshold DEG Filtering (Fold Change Analysis)
# ═══════════════════════════════════════════════════════════════
#
# A gene is a DEG only if it passes BOTH:
#   (1) Statistical: p_adj < 0.05  — controls false discovery rate
#   (2) Biological:  |log2FC| > 1  — at least 2× expression change
#
# Using statistical significance alone (Member 1's approach) can
# flag genes with tiny expression differences as significant when
# sample size is large (n=51). The FC threshold removes such
# statistically detectable but biologically irrelevant genes.

df['category'] = 'Not Significant'
df.loc[(df['log2FC'] >  FC_THRESHOLD) & (df['p_adj'] < PADJ_THRESHOLD), 'category'] = 'Upregulated'
df.loc[(df['log2FC'] < -FC_THRESHOLD) & (df['p_adj'] < PADJ_THRESHOLD), 'category'] = 'Downregulated'

# Pre-compute y-axis values; clip to avoid -log10(0) = inf
df['neg_log10_padj'] = -np.log10(df['p_adj'].clip(lower=1e-300))

n_up   = (df['category'] == 'Upregulated').sum()
n_down = (df['category'] == 'Downregulated').sum()
n_degs = n_up + n_down

print("=" * 60)
print("DEG FILTERING RESULTS")
print("=" * 60)
print(f"  Thresholds:          |log2FC| > {FC_THRESHOLD}  AND  p_adj < {PADJ_THRESHOLD}")
print(f"  Upregulated genes:   {n_up:,}")
print(f"  Downregulated genes: {n_down:,}")
print(f"  Total DEGs:          {n_degs:,}")
print(f"  Not significant:     {(df['category'] == 'Not Significant').sum():,}")
print()

# ═══════════════════════════════════════════════════════════════
# STEP 3 — Compare Against Member 1's DEG List
# ═══════════════════════════════════════════════════════════════
#
# Member 1 flags genes with p_adj < 0.05 only (column 'significant').
# Our list additionally requires |log2FC| > 1.
# By definition, our list is always a strict subset of theirs.

n_member1 = df['significant'].sum()
n_removed  = n_member1 - n_degs
pct_removed = (n_removed / n_member1 * 100) if n_member1 > 0 else 0

# Sanity check: every DEG we report must also be in Member 1's list
our_genes    = set(df.loc[df['category'] != 'Not Significant', 'gene'])
member1_genes = set(df.loc[df['significant'], 'gene'])
assert our_genes <= member1_genes, \
    "[ERROR] Some Member 2 DEGs are NOT in Member 1's list — check threshold logic."

print("=" * 60)
print("COMPARISON WITH MEMBER 1's LIST")
print("=" * 60)
print(f"  Member 1 DEGs  (p_adj < {PADJ_THRESHOLD} only):    {n_member1:,}")
print(f"  Our DEGs       (+ |log2FC| > {FC_THRESHOLD}):       {n_degs:,}")
print(f"  Removed by FC filter:               {n_removed:,}  ({pct_removed:.1f}%)")
print(f"  Subset check:                       PASSED — 100% overlap confirmed")
print()
print("  The FC filter removes statistically significant genes whose")
print("  expression differences are too small to be biologically")
print("  meaningful, particularly common with high-powered studies (n=51).")
print()

# ═══════════════════════════════════════════════════════════════
# STEP 4 — Volcano Plot
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("VOLCANO PLOT")
print("=" * 60)

top_labeled = build_volcano_plot(
    df           = df,
    fc_threshold  = FC_THRESHOLD,
    padj_threshold= PADJ_THRESHOLD,
    top_n_labels  = TOP_N_LABELS,
    output_path   = OUTPUT_FIGURE
)
print()

# ═══════════════════════════════════════════════════════════════
# STEP 5 — Export DEG List for Member 3 (GSEA Input)
# ═══════════════════════════════════════════════════════════════
#
# Member 3 will feed this CSV into GSEA / Enrichr.
# Gene names are HUGO symbols — the expected format for both tools.
# Sorted by p_adj ascending so the most significant genes are first.

os.makedirs("outputs", exist_ok=True)

degs_export = (
    df[df['category'] != 'Not Significant']
    [['gene', 'log2FC', 'p_adj', 'category']]
    .sort_values('p_adj')
    .reset_index(drop=True)
)
degs_export.to_csv(OUTPUT_DEG_CSV, index=False)

print("=" * 60)
print("EXPORTS")
print("=" * 60)
print(f"  {OUTPUT_DEG_CSV}")
print(f"    -> {len(degs_export):,} DEGs  (input for Member 3 / GSEA)")
print(f"  {OUTPUT_FIGURE}")
print(f"    -> Volcano plot  (300 DPI)")
print()

# ═══════════════════════════════════════════════════════════════
# STEP 6 — Final Summary
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Genes analysed:              {len(df):,}")
print(f"  FC threshold:                |log2FC| > {FC_THRESHOLD}  (≥ {2**FC_THRESHOLD:.0f}× change)")
print(f"  Significance threshold:      p_adj < {PADJ_THRESHOLD}  (BH / FDR)")
print()
print(f"  Member 1 DEGs (stat only):   {n_member1:,}")
print(f"  Final DEGs (dual-threshold): {n_degs:,}  ({pct_removed:.1f}% reduction)")
print(f"  [UP]   Upregulated:          {n_up:,}")
print(f"  [DOWN] Downregulated:        {n_down:,}")
print()
print(f"  Top {TOP_N_LABELS} labeled genes:")
for _, row in top_labeled[['gene', 'log2FC', 'p_adj', 'category']].iterrows():
    arrow = "[UP]  " if row['category'] == 'Upregulated' else "[DOWN]"
    print(f"    {arrow} {row['gene']:<18}  log2FC = {row['log2FC']:+.2f}   p_adj = {row['p_adj']:.2e}")
print("=" * 60)
