# ============================================================
# Data Preparation & Hypothesis Testing
# ============================================================

# ── SECTION 1: Imports ──────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os

# ── SECTION 2: Load & Verify Data ───────────────────────────

# File paths — place both files in the same folder as this script
TUMOR_FILE  = "lusc-rsem-fpkm-tcga-t_paired.txt"
NORMAL_FILE = "lusc-rsem-fpkm-tcga_paired.txt"

# Load both files (tab-separated, first column = gene names)
tumor  = pd.read_csv(TUMOR_FILE,  sep='\t', index_col=0)
normal = pd.read_csv(NORMAL_FILE, sep='\t', index_col=0)

# Verify pairing: columns (patients) must match exactly
assert list(tumor.columns) == list(normal.columns), \
    "ERROR: Sample columns do not match — data is not properly paired!"

print("=" * 55)
print("DATA LOADED SUCCESSFULLY")
print("=" * 55)
print(f"  Number of genes:    {tumor.shape[0]}")
print(f"  Number of patients: {tumor.shape[1]}")
print()

# ── SECTION 3: Preprocessing ────────────────────────────────

# --- Step 3a: Handle missing values ---
# Drop any gene that has at least one missing value in either file
genes_before = tumor.shape[0]

valid_genes = tumor.notnull().all(axis=1) & normal.notnull().all(axis=1)
tumor  = tumor[valid_genes]
normal = normal[valid_genes]

genes_after = tumor.shape[0]
dropped     = genes_before - genes_after

print("=" * 55)
print("PREPROCESSING")
print("=" * 55)
print(f"  Genes before missing-value filter: {genes_before}")
print(f"  Genes dropped (had missing values): {dropped}")
print(f"  Genes after missing-value filter:  {genes_after}")
print()

# --- Step 3b: Log2(FPKM + 1) transformation ---
# FPKM values are heavily right-skewed.
# Log2(x + 1) stabilizes variance and brings distribution
# closer to normality. The +1 avoids log(0) = -inf.
tumor_log  = np.log2(tumor  + 1)
normal_log = np.log2(normal + 1)

tumor_log_full  = tumor_log.copy()
normal_log_full = normal_log.copy()

print(f"  Log2(FPKM + 1) transformation applied.")
print()

# --- Step 3c: Filter low-expression genes ---
# Genes with near-zero expression across all samples are
# biologically uninformative. With n=51 patients, the paired
# t-test has enough power to flag even tiny random noise in
# these near-silent genes as statistically significant,
# inflating the DEG count with false discoveries.
# A gene is kept only if its mean log2(FPKM+1) expression,
# averaged across both tumor and normal samples, is >= 2.0.
# This threshold corresponds to approximately FPKM = 4,
# the widely accepted minimum for detectable expression.

MIN_EXPRESSION = 2.0

mean_expression = (tumor_log.mean(axis=1) + normal_log.mean(axis=1)) / 2
expressed_mask  = mean_expression >= MIN_EXPRESSION

genes_before_filter = tumor_log.shape[0]
tumor_log  = tumor_log[expressed_mask]
normal_log = normal_log[expressed_mask]
genes_after_filter  = tumor_log.shape[0]

print(f"  Low-expression filter (threshold = log2 {MIN_EXPRESSION}):")
print(f"    Genes removed (near-zero expression): {genes_before_filter - genes_after_filter}")
print(f"    Genes retained for testing:           {genes_after_filter}")
print()

# --- Step 3d: Plot before/after distribution ---
os.makedirs("outputs/figures", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("FPKM Distribution — Before and After Log2 Transformation",
             fontsize=14, fontweight='bold')

# --- Tumor Raw --- clip to 95th percentile, log y-axis
tumor_flat = tumor.values.flatten()
tumor_clip = np.percentile(tumor_flat, 95)
axes[0, 0].hist(tumor_flat, bins=100, color='tomato', edgecolor='none',
                range=(0, tumor_clip))
axes[0, 0].set_yscale('log')
axes[0, 0].set_title("Tumor — Raw FPKM\n(clipped to 95th percentile, log y-axis)")
axes[0, 0].set_xlabel(f"FPKM  [0 – {tumor_clip:.0f}]")
axes[0, 0].set_ylabel("Frequency (log scale)")

# --- Tumor Log-Transformed --- use FULL unfiltered log data
axes[0, 1].hist(tumor_log_full.values.flatten(), bins=100,
                color='tomato', alpha=0.8, edgecolor='none')
axes[0, 1].set_title("Tumor — Log2(FPKM + 1)")
axes[0, 1].set_xlabel("log2(FPKM + 1)")
axes[0, 1].set_ylabel("Frequency")

# --- Normal Raw --- clip to 95th percentile, log y-axis
normal_flat = normal.values.flatten()
normal_clip = np.percentile(normal_flat, 95)
axes[1, 0].hist(normal_flat, bins=100, color='steelblue', edgecolor='none',
                range=(0, normal_clip))
axes[1, 0].set_yscale('log')
axes[1, 0].set_title("Normal — Raw FPKM\n(clipped to 95th percentile, log y-axis)")
axes[1, 0].set_xlabel(f"FPKM  [0 – {normal_clip:.0f}]")
axes[1, 0].set_ylabel("Frequency (log scale)")

# --- Normal Log-Transformed --- use FULL unfiltered log data
axes[1, 1].hist(normal_log_full.values.flatten(), bins=100,
                color='steelblue', alpha=0.8, edgecolor='none')
axes[1, 1].set_title("Normal — Log2(FPKM + 1)")
axes[1, 1].set_xlabel("log2(FPKM + 1)")
axes[1, 1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("outputs/figures/fpkm_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Step 3e: check for normality ---
# The paired t-test assumes that the per-patient differences in expression (tumor - normal)
# for each gene are approximately normally distributed. We check this by performing
# a normality test (Shapiro-Wilk) on the differences for a random sample

# Compute the difference for each gene
diff = tumor_log - normal_log  # shape: (genes × patients)

# Test normality on a random sample of genes (testing all 20k is slow)
sample_genes = diff.sample(10000, random_state=42)

normal_count = 0
for gene, row in sample_genes.iterrows():
    stat, p = stats.shapiro(row.values)
    if p > 0.05:
        normal_count += 1

print("=" * 55)
print("NORMALITY CHECK — SHAPIRO-WILK")
print("=" * 55)
print(f"  Genes passing normality (p > 0.05): {normal_count}/10000 ({normal_count/100:.1f}%)")
print()

# ── SECTION 4: Hypothesis Testing — Paired T-Test ───────────
#
# Justification:
#   Data is paired — each tumor sample comes from the same
#   patient as the corresponding normal sample. The paired
#   t-test accounts for inter-patient variability by working
#   on the per-patient differences (tumor - normal)
#
# Test setup:
#   H0: mean difference in expression = 0  (not a DEG)
#   H1: mean difference in expression ≠ 0  (is a DEG)
#   Significance level: α = 0.05 (before correction)
#   Test type: two-tailed (gene can be up OR down regulated)

print("=" * 55)
print("HYPOTHESIS TESTING — PAIRED T-TEST")
print("=" * 55)

n_genes    = tumor_log.shape[0]
gene_names = tumor_log.index.tolist()

t_stats  = np.zeros(n_genes)
p_values = np.zeros(n_genes)

for i, gene in enumerate(gene_names):
    t, p = stats.ttest_rel(
        tumor_log.loc[gene].values,    # tumor expression for this gene
        normal_log.loc[gene].values    # matched normal expression
    )
    t_stats[i]  = t
    p_values[i] = p

# Collect into a dataframe
results = pd.DataFrame({
    'gene':    gene_names,
    't_stat':  t_stats,
    'p_value': p_values
})

print(f"  Paired t-test applied to {n_genes} genes.")
print(f"  Genes with raw p-value < 0.05: {(p_values < 0.05).sum()}")
print()

# ── SECTION 5: Multiple Testing Correction — Benjamini-Hochberg ──
#
# Problem:
#   Running ~20,000 simultaneous tests means ~1,000 false
#   positives are expected by chance at α = 0.05.
#
# Solution — Benjamini-Hochberg (BH) / FDR correction:
#   Controls the False Discovery Rate (FDR): the expected
#   proportion of false positives among all rejected tests.
#   FDR = 0.05 means at most 5% of called DEGs are expected
#   to be false positives. This is the standard threshold
#   used in genomics studies.
#
# How BH works:
#   1. Sort p-values: p(1) ≤ p(2) ≤ ... ≤ p(m)
#   2. For rank k: BH critical value = (k / m) × Q  where Q = 0.05
#   3. Find largest k where p(k) ≤ BH critical value → k*
#   4. All genes with rank ≤ k* are declared significant

reject, p_adj, _, _ = multipletests(
    results['p_value'],
    alpha=0.05,
    method='fdr_bh'    # Benjamini-Hochberg
)

results['p_adj']       = p_adj
results['significant'] = reject

# Compute log2 fold change and add to results
# (required by teammates for volcano plot)
log2fc = tumor_log.mean(axis=1).values - normal_log.mean(axis=1).values
results['log2FC'] = log2fc

n_sig = reject.sum()

print("=" * 55)
print("MULTIPLE TESTING CORRECTION — BENJAMINI-HOCHBERG")
print("=" * 55)
print(f"  Total genes tested:               {n_genes}")
print(f"  Significant after BH (FDR < 0.05): {n_sig}")
print()

# ── SECTION 6: Export Results ───────────────────────────────

os.makedirs("outputs", exist_ok=True)

# 1. Full results for ALL genes — teammates need this for volcano plot
results.to_csv("outputs/all_genes_results.csv", index=False)

# 2. Significant DEGs only
degs = results[results['significant']].copy()
degs = degs.sort_values('p_adj')
degs['regulation'] = degs['log2FC'].apply(lambda x: 'UP' if x > 0 else 'DOWN')
degs.to_csv("outputs/DEGs_significant.csv", index=False)

print("=" * 55)
print("RESULTS EXPORTED")
print("=" * 55)
print(f"  outputs/all_genes_results.csv  — {len(results)} genes (for teammates)")
print(f"  outputs/DEGs_significant.csv   — {len(degs)} significant DEGs")
print()

# ── SECTION 7: Summary ──────────────────────────────────────

up_count   = (degs['regulation'] == 'UP').sum()
down_count = (degs['regulation'] == 'DOWN').sum()

print("=" * 55)
print("FINAL SUMMARY")
print("=" * 55)
print(f"  Total genes loaded:          {genes_before}")
print(f"  Genes after missing filter:  {genes_after}")
print(f"  Genes after expression filter: {genes_after_filter}")
print(f"  Genes tested:                {n_genes}")
print(f"  Test used:                   Paired t-test (two-tailed)")
print(f"  Correction method:           Benjamini-Hochberg (FDR)")
print(f"  FDR threshold:               0.05")
print(f"  Significant DEGs found:      {n_sig}")
print(f"    ↑ Upregulated:             {up_count}")
print(f"    ↓ Downregulated:           {down_count}")
print(f"  Distribution plot saved:     outputs/figures/fpkm_distribution.png")
print("=" * 55)