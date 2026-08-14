# ===== run_all.py =====
"""
Batch experiment runner for comparing original P-ACO vs P-ACO+PLS.

This script runs both algorithm versions across all instances (3 scales × 5 seeds)
and generates comparison tables and plots.

Usage:
    python run_all.py
"""

import sys
import os
import pandas as pd
import time
import importlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from config import SEEDS, SCALE_CONFIGS

# Add current directory to path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_config(scale):
    """Get configuration for the given scale."""
    if scale not in SCALE_CONFIGS:
        raise ValueError(f"Configuration not found for scale {scale}")
    return SCALE_CONFIGS[scale]


def build_instance_with_params(cust_n, e_window, l_window):
    """
    Generate an instance with custom parameters.

    Args:
        cust_n: Number of customers
        e_window: [min, max] for earliest service time
        l_window: [min, max] for latest service time

    Returns:
        nodes, dist_mat, demand_list, e_list, l_list
    """
    import numpy as np

    # Coordinate scale based on problem size
    if cust_n <= 25:
        scale = 5
    elif cust_n <= 50:
        scale = 10
    else:
        scale = 15

    depot = np.array([[0, 0]])
    custs = np.random.uniform(-scale, scale, (cust_n, 2))
    nodes = np.vstack([depot, custs])
    n = len(nodes)
    dist = np.linalg.norm(nodes[:, None] - nodes[None, :], axis=-1)
    demand = np.random.randint(1, 10, size=cust_n)
    e = np.random.randint(e_window[0], e_window[1], cust_n)
    l = np.random.randint(l_window[0], l_window[1], cust_n)
    return nodes, dist, demand, e, l


def run_on_instances(version_name, module_name):
    """
    Run the specified algorithm version on all instances.

    Args:
        version_name: Display name for the version (e.g., 'original', 'with_pls')
        module_name: Python module name to import

    Returns:
        DataFrame with all results
    """
    results = []

    # Try to import the module
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Cannot import {module_name}: {e}")
        return pd.DataFrame()

    total = len(SEEDS) * len(SCALE_CONFIGS)
    count = 0

    for scale, config in SCALE_CONFIGS.items():
        for seed in SEEDS:
            count += 1
            print(f"[{count}/{total}] {version_name}: {scale} customers, seed={seed}")

            try:
                # Set random seed for reproducibility
                import numpy as np
                import random
                np.random.seed(seed)
                random.seed(seed)

                # Generate instance
                nodes, dist_mat, demand_list, e_list, l_list = build_instance_with_params(
                    scale,
                    e_window=config['e_window'],
                    l_window=config['l_window']
                )

                # Run the algorithm
                start = time.time()
                result = module.run_algorithm(
                    scale=scale,
                    seed=seed,
                    endurance=config['endurance'],
                    e_window=config['e_window'],
                    l_window=config['l_window']
                )
                elapsed = time.time() - start

                results.append({
                    'scale': scale,
                    'seed': seed,
                    'version': version_name,

                    # Combined best solution
                    'combined_cost': result['combined_cost'],
                    'combined_delay': result['combined_delay'],
                    'combined_objective': result['combined_objective'],

                    # Cost best solution
                    'cost_min_cost': result['cost_min_cost'],
                    'cost_min_delay': result['cost_min_delay'],
                    'cost_min_objective': result['cost_min_objective'],

                    # Delay best solution
                    'delay_min_cost': result['delay_min_cost'],
                    'delay_min_delay': result['delay_min_delay'],
                    'delay_min_objective': result['delay_min_objective'],

                    'pareto_size': result['pareto_size'],
                    'runtime': elapsed,
                })

                print(f"    Combined: Cost={result['combined_cost']:.2f}, "
                      f"Delay={result['combined_delay']:.2f}, "
                      f"Obj={result['combined_objective']:.2f}")
                print(f"    Cost Min: Cost={result['cost_min_cost']:.2f}, "
                      f"Delay={result['cost_min_delay']:.2f}, "
                      f"Obj={result['cost_min_objective']:.2f}")
                print(f"    Delay Min: Cost={result['delay_min_cost']:.2f}, "
                      f"Delay={result['delay_min_delay']:.2f}, "
                      f"Obj={result['delay_min_objective']:.2f}")
                print(f"    Pareto Points: {result['pareto_size']}")

            except Exception as e:
                print(f"    Failed: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'scale': scale,
                    'seed': seed,
                    'version': version_name,
                    'combined_objective': None,
                    'pareto_size': None,
                    'runtime': None,
                })

    return pd.DataFrame(results)


def generate_comparison_plots(df, save_dir="compare_results"):
    """
    Generate comparison plots for the two algorithm versions.

    Plots:
    1. Boxplot: Objective distribution by scale
    2. Bar chart: PLS improvement rate
    3. Error bar: Performance comparison across scales
    4. Bar chart: Pareto front size comparison
    """
    if df.empty:
        print("No data available")
        return

    os.makedirs(save_dir, exist_ok=True)

    df_valid = df[df['combined_objective'].notna()].copy()
    if df_valid.empty:
        print("No valid data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    scales = sorted(df_valid['scale'].unique())
    versions = df_valid['version'].unique()

    # --- Plot 1: Boxplot ---
    ax = axes[0, 0]
    data_list = []
    labels = []
    positions = []
    pos = 0
    for scale in scales:
        for version in versions:
            data = df_valid[(df_valid['scale'] == scale) &
                           (df_valid['version'] == version)]['combined_objective'].dropna()
            if len(data) > 0:
                data_list.append(data)
                labels.append(f'{scale} cust.\n{version}')
                positions.append(pos)
                pos += 1
        pos += 1

    if data_list:
        bp = ax.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True)
        colors = ['lightblue', 'lightgreen'] * len(scales)
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Objective Value')
        ax.set_title('Objective Distribution by Scale')
        ax.grid(True, axis='y')

    # --- Plot 2: Improvement Rate ---
    ax = axes[0, 1]
    improvements = []
    scale_labels = []
    for scale in scales:
        orig_data = df_valid[(df_valid['scale'] == scale) &
                            (df_valid['version'] == 'original')]['combined_objective']
        pls_data = df_valid[(df_valid['scale'] == scale) &
                           (df_valid['version'] == 'with_pls')]['combined_objective']
        if len(orig_data) > 0 and len(pls_data) > 0:
            orig_mean = orig_data.mean()
            pls_mean = pls_data.mean()
            if orig_mean > 0:
                improvements.append((orig_mean - pls_mean) / orig_mean * 100)
                scale_labels.append(f'{scale} cust.')

    if improvements:
        bars = ax.bar(scale_labels, improvements,
                      color=['steelblue', 'mediumseagreen', 'coral'])
        ax.set_ylabel('Improvement Rate (%)')
        ax.set_title('PLS Improvement vs Original')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        for bar, val in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'+{val:.1f}%', ha='center', fontweight='bold')
        ax.grid(True, axis='y')

    # --- Plot 3: Error Bar Plot ---
    ax = axes[1, 0]
    for version, color, marker in [('original', 'blue', 'o'),
                                   ('with_pls', 'red', 's')]:
        data = df_valid[df_valid['version'] == version]
        means = data.groupby('scale')['combined_objective'].mean()
        stds = data.groupby('scale')['combined_objective'].std()
        if len(means) > 0:
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        capsize=5, marker=marker, color=color,
                        label=version, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Customers')
    ax.set_ylabel('Avg. Objective Value')
    ax.legend()
    ax.set_title('Performance Comparison Across Scales')
    ax.grid(True)

    # --- Plot 4: Pareto Size ---
    ax = axes[1, 1]
    for version, color in [('original', 'blue'), ('with_pls', 'red')]:
        data = df_valid[df_valid['version'] == version]
        means = data.groupby('scale')['pareto_size'].mean()
        stds = data.groupby('scale')['pareto_size'].std()
        if len(means) > 0:
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        capsize=5, marker='o', color=color,
                        label=version, linewidth=2)
    ax.set_xlabel('Number of Customers')
    ax.set_ylabel('Avg. Pareto Points')
    ax.legend()
    ax.set_title('Pareto Front Size Comparison')
    ax.grid(True)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"comparison_plots_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparison plots saved: {save_path}")
    return save_path


def print_comparison_table(df, save_dir="compare_results"):
    """
    Print and save comparison summary table.
    """
    if df.empty:
        print("No data available")
        return

    df_valid = df[df['combined_objective'].notna()].copy()
    if df_valid.empty:
        print("No valid data")
        return

    print("\n" + "="*70)
    print("Detailed Comparison Results")
    print("="*70)

    summary_data = []
    for scale in sorted(df_valid['scale'].unique()):
        for version in ['original', 'with_pls']:
            data = df_valid[(df_valid['scale'] == scale) &
                           (df_valid['version'] == version)]
            if len(data) > 0:
                row = {
                    'Scale': scale,
                    'Version': version,
                    'Avg Objective': data['combined_objective'].mean(),
                    'Std Objective': data['combined_objective'].std(),
                    'Avg Cost': data['combined_cost'].mean(),
                    'Avg Delay': data['combined_delay'].mean(),
                    'Avg Pareto Points': data['pareto_size'].mean(),
                    'Avg Runtime (s)': data['runtime'].mean(),
                }
                summary_data.append(row)

        # Calculate improvement rate
        orig = df_valid[(df_valid['scale'] == scale) &
                       (df_valid['version'] == 'original')]
        pls = df_valid[(df_valid['scale'] == scale) &
                      (df_valid['version'] == 'with_pls')]
        if len(orig) > 0 and len(pls) > 0:
            orig_mean = orig['combined_objective'].mean()
            pls_mean = pls['combined_objective'].mean()
            if orig_mean > 0:
                improvement = (orig_mean - pls_mean) / orig_mean * 100
                print(f"\n[{scale} customers] PLS improvement: {improvement:.2f}%")

    # Save to CSV
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"comparison_table_{timestamp}.csv")
        df_summary.to_csv(save_path, index=False)
        print(f"\nComparison table saved: {save_path}")


# ====================== Main ======================

if __name__ == "__main__":
    # Configure the two versions to compare
    versions = [
        ('with_pls', 'P-ACO_PLS'),        # Module name (without .py)
        ('original', 'original_P-ACO'),   # Module name (without .py)
    ]

    # Create result directories
    os.makedirs("compare_results", exist_ok=True)
    for name, _ in versions:
        os.makedirs(f"compare_results/{name}", exist_ok=True)

    all_dfs = []

    for name, module in versions:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        df = run_on_instances(name, module)
        all_dfs.append(df)

        if not df.empty:
            df.to_csv(f"compare_results/{name}/results.csv", index=False)
            print(f"{name} results saved to compare_results/{name}/results.csv")

    # Merge and compare
    valid_dfs = [df for df in all_dfs if not df.empty]
    if valid_dfs:
        combined = pd.concat(valid_dfs, ignore_index=True)

        combined.to_csv("compare_results/all_results.csv", index=False)
        print("\nAll results saved to compare_results/all_results.csv")

        print_comparison_table(combined, save_dir="compare_results")
        generate_comparison_plots(combined, save_dir="compare_results")

    else:
        print("\nNo successful results to aggregate")