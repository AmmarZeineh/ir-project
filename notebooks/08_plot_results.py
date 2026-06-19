import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def plot_dataset_results(dataset_name, csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"⚠️ CSV file for {dataset_name} not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    df.set_index('Model', inplace=True)
    
    metrics = ['MAP', 'nDCG@10', 'P@10', 'Recall']
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(df.index))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, df['MAP'], width, label='MAP', color='#38bdf8')
    rects2 = ax.bar(x + width/2, df['nDCG@10'], width, label='nDCG@10', color='#818cf8')
    
    ax.set_ylabel('Score')
    ax.set_title(f'Evaluation Metrics (MAP & nDCG@10) - {dataset_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')
    
    fig.tight_layout()
    plot_path = os.path.join(output_dir, f'{dataset_name}_map_ndcg.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved plot to {plot_path}")
    
    if 'BM25' in df.index and 'BM25 (Refined)' in df.index:
        plt.figure(figsize=(8, 5))
        compare_df = df.loc[['BM25', 'BM25 (Refined)'], ['MAP', 'nDCG@10']]
        
        ax = compare_df.plot(kind='bar', figsize=(8, 5), color=['#38bdf8', '#818cf8'])
        plt.title(f'Query Refinement Impact - {dataset_name}')
        plt.ylabel('Score')
        plt.xticks(rotation=0)
        plt.legend(loc='lower right')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
            
        plt.tight_layout()
        ref_plot_path = os.path.join(output_dir, f'{dataset_name}_refinement_impact.png')
        plt.savefig(ref_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved refinement comparison plot to {ref_plot_path}")


if __name__ == "__main__":
    print("📊 Generating Bar Charts for Evaluation Results...")
    
    plot_dataset_results(
        dataset_name="Quora", 
        csv_path="data/processed/quora/evaluation_results.csv",
        output_dir="data/processed/quora/plots"
    )
    
    plot_dataset_results(
        dataset_name="Touche-2020", 
        csv_path="data/processed/touche/evaluation_results.csv",
        output_dir="data/processed/touche/plots"
    )
    
    print("\n✨ Done! You can now include these PNG images in your report.")
