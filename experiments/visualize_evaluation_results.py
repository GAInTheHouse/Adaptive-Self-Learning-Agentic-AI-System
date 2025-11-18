"""
Visualization script for evaluation framework results.
Generates charts and plots for evaluation metrics.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_evaluation_results(results_path: str = "experiments/evaluation_outputs/evaluation_summary.json"):
    """Load evaluation results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_wer_cer_comparison(summary: dict, output_path: str):
    """Plot WER and CER comparison across datasets."""
    per_dataset = summary.get('per_dataset_metrics', {})
    
    if not per_dataset:
        print("No per-dataset metrics found")
        return
    
    datasets = list(per_dataset.keys())
    wers = [per_dataset[d]['wer'] for d in datasets]
    cers = [per_dataset[d]['cer'] for d in datasets]
    
    x = range(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots()
    ax.bar([i - width/2 for i in x], wers, width, label='WER', alpha=0.8)
    ax.bar([i + width/2 for i in x], cers, width, label='CER', alpha=0.8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Error Rate')
    ax.set_title('WER and CER Comparison Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to {output_path}")
    plt.close()


def plot_error_distribution(detailed_results_path: str, output_path: str):
    """Plot distribution of WER across samples."""
    with open(detailed_results_path, 'r') as f:
        results = json.load(f)
    
    all_wers = []
    for dataset_name, dataset_results in results.get('detailed_results', {}).items():
        if 'error_analysis' in dataset_results:
            worst_errors = dataset_results['error_analysis'].get('worst_errors', [])
            wers = [e['wer'] for e in worst_errors]
            all_wers.extend(wers)
    
    if not all_wers:
        print("No error data found for plotting")
        return
    
    plt.figure()
    plt.hist(all_wers, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Word Error Rate (WER)')
    plt.ylabel('Frequency')
    plt.title('Distribution of WER Across Evaluation Samples')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to {output_path}")
    plt.close()


def create_evaluation_dashboard(summary_path: str, output_path: str):
    """Create comprehensive evaluation dashboard."""
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. WER/CER comparison
    per_dataset = summary.get('per_dataset_metrics', {})
    if per_dataset:
        datasets = list(per_dataset.keys())
        wers = [per_dataset[d]['wer'] for d in datasets]
        cers = [per_dataset[d]['cer'] for d in datasets]
        
        axes[0, 0].bar(datasets, wers, alpha=0.7, label='WER')
        axes[0, 0].set_title('Word Error Rate by Dataset')
        axes[0, 0].set_ylabel('WER')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].bar(datasets, cers, alpha=0.7, color='orange', label='CER')
        axes[0, 1].set_title('Character Error Rate by Dataset')
        axes[0, 1].set_ylabel('CER')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
    
    # 2. Overall metrics summary
    overall = summary.get('overall_metrics', {})
    if overall.get('mean_wer') is not None:
        metrics_text = f"""
        Model: {summary.get('model', 'N/A')}
        Total Samples: {summary.get('total_samples_evaluated', 0)}
        
        Mean WER: {overall['mean_wer']:.4f} ± {overall.get('std_wer', 0):.4f}
        Mean CER: {overall['mean_cer']:.4f} ± {overall.get('std_cer', 0):.4f}
        Best WER: {overall.get('best_wer', 0):.4f}
        Worst WER: {overall.get('worst_wer', 0):.4f}
        """
        axes[1, 0].text(0.1, 0.5, metrics_text, fontsize=12, 
                        verticalalignment='center', family='monospace')
        axes[1, 0].set_title('Overall Metrics Summary')
        axes[1, 0].axis('off')
    
    # 3. Sample count by dataset
    if per_dataset:
        datasets = list(per_dataset.keys())
        sample_counts = [per_dataset[d]['num_samples'] for d in datasets]
        
        axes[1, 1].bar(datasets, sample_counts, alpha=0.7, color='green')
        axes[1, 1].set_title('Samples Evaluated per Dataset')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('STT Model Evaluation Dashboard', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved dashboard to {output_path}")
    plt.close()


def main():
    """Main visualization execution."""
    print("="*70)
    print("EVALUATION RESULTS VISUALIZATION")
    print("="*70)
    
    results_dir = Path("experiments/evaluation_outputs")
    summary_path = results_dir / "evaluation_summary.json"
    detailed_path = results_dir / "evaluation_report.json"
    
    if not summary_path.exists():
        print(f"❌ Evaluation results not found at {summary_path}")
        print("   Run kavya_evaluation_framework.py first")
        return
    
    # Load results
    summary = load_evaluation_results(str(summary_path))
    
    # Create visualizations
    output_dir = results_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    print("\n📊 Generating visualizations...")
    
    # WER/CER comparison
    plot_wer_cer_comparison(
        summary,
        str(output_dir / "wer_cer_comparison.png")
    )
    
    # Error distribution
    if detailed_path.exists():
        plot_error_distribution(
            str(detailed_path),
            str(output_dir / "error_distribution.png")
        )
    
    # Dashboard
    create_evaluation_dashboard(
        str(summary_path),
        str(output_dir / "evaluation_dashboard.png")
    )
    
    print(f"\n✅ All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
