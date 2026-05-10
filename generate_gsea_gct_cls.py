import pandas as pd
import numpy as np
import os

# ── Configuration ─────────────────────────────────────────────
TUMOR_FILE  = "lusc-rsem-fpkm-tcga-t_paired.txt"
NORMAL_FILE = "lusc-rsem-fpkm-tcga_paired.txt"
DEG_FILE    = "outputs/DEGs_volcano_filtered.csv"

OUTPUT_GCT  = "outputs/LUSC_DEGs.gct"
OUTPUT_CLS  = "outputs/LUSC_Phenotypes.cls"

def generate_gsea_files():
    print("=" * 60)
    print("GENERATING GSEA INPUT FILES (GCT & CLS) - FIXED COLUMNS")
    print("=" * 60)

    # 1. Load the list of DEGs identified by the volcano plot
    if not os.path.exists(DEG_FILE):
        print(f"[ERROR] '{DEG_FILE}' not found. Run analysis scripts first.")
        return
    degs_df = pd.read_csv(DEG_FILE)
    deg_list = degs_df['gene'].tolist()

    # 2. Load raw expression data
    tumor  = pd.read_csv(TUMOR_FILE,  sep='\t', index_col=0)
    normal = pd.read_csv(NORMAL_FILE, sep='\t', index_col=0)

    # 3. Extract description (Entrez ID) and drop from both to prevent duplicates
    description = tumor['Entrez_Gene_Id'].copy()
    tumor = tumor.drop(columns=['Entrez_Gene_Id'])
    normal = normal.drop(columns=['Entrez_Gene_Id'])

    # 4. Rename columns to ensure they are unique
    # Since patients are paired, the column names are identical in both files.
    tumor.columns = [str(col) + "_Tumor" for col in tumor.columns]
    normal.columns = [str(col) + "_Normal" for col in normal.columns]

    # 5. Combine and Filter to DEGs only
    combined = pd.concat([tumor, normal], axis=1)
    
    # Filter to only include the genes identified as DEGs
    gct_data = combined.loc[combined.index.intersection(deg_list)].copy()
    
    # Add the Description column
    gct_data.insert(0, 'Description', description.loc[gct_data.index])
    gct_data.index.name = 'Name'
    
    # 6. Create GCT File
    with open(OUTPUT_GCT, 'w', newline='') as f:
        f.write("#1.2\n")
        # gct_data columns currently include 'Description' + all sample columns
        # The number of samples is len(columns) - 1
        num_samples = len(gct_data.columns) - 1
        f.write(f"{len(gct_data)}\t{num_samples}\n")
        gct_data.to_csv(f, sep='\t')

    print(f"  Created GCT: {OUTPUT_GCT} ({len(gct_data)} genes x {num_samples} samples)")

    # 7. Create CLS File
    num_tumor = len(tumor.columns)
    num_normal = len(normal.columns)
    total_samples = num_tumor + num_normal

    with open(OUTPUT_CLS, 'w', newline='') as f:
        f.write(f"{total_samples} 2 1\n")
        f.write("# Tumor Normal\n")
        labels = (["Tumor"] * num_tumor) + (["Normal"] * num_normal)
        f.write(" ".join(labels) + "\n")

    print(f"  Created CLS: {OUTPUT_CLS}")
    print("\n[SUCCESS] Fixed duplicate column names. Files are ready for GSEA.")
    print("=" * 60)

if __name__ == "__main__":
    generate_gsea_files()
