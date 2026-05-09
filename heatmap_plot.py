import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def build_heatmap(tumor_df, normal_df, top_genes, output_path):
    """
    Generate a heatmap of expression for the top genes.
    
    Parameters
    ----------
    tumor_df    : pd.DataFrame - raw tumor expression data
    normal_df   : pd.DataFrame - raw normal expression data
    top_genes   : list of str  - list of gene names to plot
    output_path : str          - path to save the heatmap
    """
    # 1. Filter expression data for the top genes
    tumor_sub = tumor_df.loc[tumor_df.index.isin(top_genes)].copy()
    normal_sub = normal_df.loc[normal_df.index.isin(top_genes)].copy()
    
    # 2. Reorder to match the top_genes list order exactly
    tumor_sub = tumor_sub.loc[top_genes]
    normal_sub = normal_sub.loc[top_genes]
    
    # 3. Log2(x+1) transformation
    tumor_log = np.log2(tumor_sub + 1)
    normal_log = np.log2(normal_sub + 1)
    
    # 4. Combine into a single matrix: Normal samples first, then Tumor samples
    normal_log.columns = [f"N_{c}" for c in normal_log.columns]
    tumor_log.columns = [f"T_{c}" for c in tumor_log.columns]
    combined_log = pd.concat([normal_log, tumor_log], axis=1)
    
    # 5. Z-score normalization per gene (row-wise)
    # Z = (x - mean) / std
    mean = combined_log.mean(axis=1)
    std = combined_log.std(axis=1)
    z_scores = combined_log.subtract(mean, axis=0).divide(std, axis=0)
    
    # 6. Plotting
    plt.figure(figsize=(16, 10))
    # We use a diverging colormap suitable for Z-scores
    ax = sns.heatmap(
        z_scores,
        cmap='vlag',
        center=0,
        vmin=-3,
        vmax=3,
        cbar_kws={'label': 'Z-score (Log2 Expression)'},
        yticklabels=True,
        xticklabels=False # Too many samples to show individual names cleanly
    )
    
    # Add vertical line to separate Normal and Tumor
    plt.axvline(x=normal_log.shape[1], color='black', linewidth=2)
    
    # Add column annotations for Normal vs Tumor
    ax.text(normal_log.shape[1] / 2, -0.5, 'Normal', size=14, ha='center', va='bottom', fontweight='bold')
    ax.text(normal_log.shape[1] + tumor_log.shape[1] / 2, -0.5, 'Tumor', size=14, ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Expression Heatmap of Top {len(top_genes)} DEGs (by Absolute Fold Change)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Genes', fontsize=14)
    plt.xlabel('Samples', fontsize=14)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Heatmap saved: {output_path}")
