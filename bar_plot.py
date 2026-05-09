# ============================================================
# bar_plot.py
# Module — Bar Plot for DEG Counts Comparison
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import os

def build_bar_plot(n1_up, n1_down, n2_up, n2_down, output_path):
    """
    Draw and save a bar chart comparing DEG counts between Statistical and Dual-Threshold criteria.
    
    Parameters
    ----------
    n1_up       : int - Statistical Upregulated count (p_adj < 0.05, log2FC > 0)
    n1_down     : int - Statistical Downregulated count (p_adj < 0.05, log2FC < 0)
    n2_up       : int - Dual-Threshold Upregulated count (p_adj < 0.05, log2FC > 1)
    n2_down     : int - Dual-Threshold Downregulated count (p_adj < 0.05, log2FC < -1)
    output_path : str - file path to save the PNG figure
    """
    labels = ['Statistical Significance Only\n(p_adj < 0.05)', 'Dual-Threshold Filtering\n(p_adj < 0.05 & |log2FC| > 1)']
    
    up_counts = [n1_up, n2_up]
    down_counts = [n1_down, n2_down]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    
    rects1 = ax.bar(x - width/2, up_counts, width, label='Upregulated', color='#D6604D', edgecolor='#a8382a', zorder=3)
    rects2 = ax.bar(x + width/2, down_counts, width, label='Downregulated', color='#2166AC', edgecolor='#1a4f88', zorder=3)
    
    ax.set_ylabel('Number of Genes', fontsize=12, labelpad=8)
    ax.set_title('DEG Counts Comparison: Statistical vs. Dual-Threshold Filtering', fontsize=13, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.85, edgecolor='#CCCCCC')
    
    ax.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.5, color='#CCCCCC', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    
    # Add count labels on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:,}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9.5, color='#333333')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    # Add a summary text box
    stats_text = (
        f"Statistical-Only Total: {n1_up + n1_down:,}\n"
        f"Dual-Threshold Total: {n2_up + n2_down:,}\n"
        f"Reduction: {((n1_up + n1_down) - (n2_up + n2_down)) / (n1_up + n1_down) * 100:.1f}%" if (n1_up + n1_down) > 0 else "Reduction: 0%"
    )
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes, fontsize=9.5,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#CCCCCC', alpha=0.9)
    )
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"  Bar plot saved: {output_path}")
