# ===== run_all.py =====
import sys
import os
import pickle
import pandas as pd
import time
import importlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from config import SEEDS, SCALE_CONFIGS

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_config(scale):
    """获取对应规模的配置"""
    if scale not in SCALE_CONFIGS:
        raise ValueError(f"未找到规模 {scale} 的配置")
    return SCALE_CONFIGS[scale]

def build_instance_with_params(cust_n, e_window, l_window):
    """带参数的算例生成 - 直接复制自版本文件"""
    import numpy as np
    
    # 根据规模确定坐标范围
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
    对15个实例运行指定版本
    """
    results = []
    
    # 尝试导入模块
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"❌ 无法导入 {module_name}: {e}")
        return pd.DataFrame()
    
    total = len(SEEDS) * len(SCALE_CONFIGS)
    count = 0
    
    for scale, config in SCALE_CONFIGS.items():
        for seed in SEEDS:
            count += 1
            print(f"[{count}/{total}] {version_name}: {scale}客户, seed={seed}")
            
            try:
                # 设置随机种子
                import numpy as np
                import random
                np.random.seed(seed)
                random.seed(seed)
                
                # 生成实例
                nodes, dist_mat, demand_list, e_list, l_list = build_instance_with_params(
                    scale,
                    e_window=config['e_window'],
                    l_window=config['l_window']
                )
                
                # 调用版本的 run_algorithm
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
                    
                    # ===== 综合最小 =====
                    'combined_cost': result['combined_cost'],
                    'combined_delay': result['combined_delay'],
                    'combined_objective': result['combined_objective'],
                    
                    # ===== 成本最小 =====
                    'cost_min_cost': result['cost_min_cost'],
                    'cost_min_delay': result['cost_min_delay'],
                    'cost_min_objective': result['cost_min_objective'],
                    
                    # ===== 延迟最小 =====
                    'delay_min_cost': result['delay_min_cost'],
                    'delay_min_delay': result['delay_min_delay'],
                    'delay_min_objective': result['delay_min_objective'],
                    
                    'pareto_size': result['pareto_size'],
                    'runtime': elapsed,
                })
                # print(f"    ✅ 综合={result['combined_objective']:.2f}, 点数={result['pareto_size']}")
                print(f"    ✅ 综合最小: 成本={result['combined_cost']:.2f}, 延迟={result['combined_delay']:.2f}, 综合={result['combined_objective']:.2f}")
                print(f"       成本最小: 成本={result['cost_min_cost']:.2f}, 延迟={result['cost_min_delay']:.2f}, 综合={result['cost_min_objective']:.2f}")
                print(f"       延迟最小: 成本={result['delay_min_cost']:.2f}, 延迟={result['delay_min_delay']:.2f}, 综合={result['delay_min_objective']:.2f}")
                print(f"       帕累托点数: {result['pareto_size']}")
                
            except Exception as e:
                print(f"    ❌ 失败: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'scale': scale,
                    'seed': seed,
                    'version': version_name,
                    'best_objective': None,
                    'pareto_size': None,
                    'runtime': None,
                })
    
    return pd.DataFrame(results)

def generate_comparison_plots(df, save_dir="compare_results"):
    """生成对比图表（英文版）"""
    if df.empty:
        print("❌ No data")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    df_valid = df[df['combined_objective'].notna()].copy()
    if df_valid.empty:
        print("❌ No valid data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 获取所有规模
    scales = sorted(df_valid['scale'].unique())
    versions = df_valid['version'].unique()
    
    # 1. 箱线图 - Boxplot
    ax = axes[0, 0]
    data_list = []
    labels = []
    positions = []
    pos = 0
    for scale in scales:
        for version in versions:
            data = df_valid[(df_valid['scale'] == scale) & (df_valid['version'] == version)]['combined_objective'].dropna()
            if len(data) > 0:
                data_list.append(data)
                labels.append(f'{scale} cust.\n{version}')
                positions.append(pos)
                pos += 1
        pos += 1  # 间隔
    
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
    
    # 2. 改进率柱状图 - Improvement Rate
    ax = axes[0, 1]
    improvements = []
    scale_labels = []
    for scale in scales:
        orig_data = df_valid[(df_valid['scale'] == scale) & (df_valid['version'] == 'original')]['combined_objective']
        pls_data = df_valid[(df_valid['scale'] == scale) & (df_valid['version'] == 'with_pls')]['combined_objective']
        if len(orig_data) > 0 and len(pls_data) > 0:
            orig_mean = orig_data.mean()
            pls_mean = pls_data.mean()
            if orig_mean > 0:
                improvements.append((orig_mean - pls_mean) / orig_mean * 100)
                scale_labels.append(f'{scale} cust.')
    
    if improvements:
        bars = ax.bar(scale_labels, improvements, color=['steelblue', 'mediumseagreen', 'coral'])
        ax.set_ylabel('Improvement Rate (%)')
        ax.set_title('PLS Improvement vs Original')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        for bar, val in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'+{val:.1f}%', ha='center', fontweight='bold')
        ax.grid(True, axis='y')
    
    # 3. 误差棒线图 - Error Bar Plot
    ax = axes[1, 0]
    for version, color, marker in [('original', 'blue', 'o'), ('with_pls', 'red', 's')]:
        data = df_valid[df_valid['version'] == version]
        means = data.groupby('scale')['combined_objective'].mean()
        stds = data.groupby('scale')['combined_objective'].std()
        if len(means) > 0:
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        capsize=5, marker=marker, color=color, label=version, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Customers')
    ax.set_ylabel('Avg. Objective Value')
    ax.legend()
    ax.set_title('Performance Comparison Across Scales')
    ax.grid(True)
    
    # 4. 帕累托点数对比 - Pareto Points
    ax = axes[1, 1]
    for version, color in [('original', 'blue'), ('with_pls', 'red')]:
        data = df_valid[df_valid['version'] == version]
        means = data.groupby('scale')['pareto_size'].mean()
        stds = data.groupby('scale')['pareto_size'].std()
        if len(means) > 0:
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        capsize=5, marker='o', color=color, label=version, linewidth=2)
    ax.set_xlabel('Number of Customers')
    ax.set_ylabel('Avg. Pareto Points')
    ax.legend()
    ax.set_title('Pareto Front Size Comparison')
    ax.grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"comparison_plots_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Comparison plots saved: {save_path}")
    return save_path

def print_comparison_table(df, save_dir="compare_results"):
    """打印并保存对比表格"""
    if df.empty:
        print("❌ 没有数据")
        return
    
    df_valid = df[df['combined_objective'].notna()].copy()
    if df_valid.empty:
        print("❌ 没有有效数据")
        return
    
    print("\n" + "="*70)
    print("详细对比结果")
    print("="*70)
    
    summary_data = []
    for scale in sorted(df_valid['scale'].unique()):
        for version in ['original', 'with_pls']:
            data = df_valid[(df_valid['scale'] == scale) & (df_valid['version'] == version)]
            if len(data) > 0:
                row = {
                    '规模': scale,
                    '版本': version,
                    '平均综合目标': data['combined_objective'].mean(),
                    '综合目标标准差': data['combined_objective'].std(),
                    '平均成本': data['combined_cost'].mean(),
                    '平均延迟': data['combined_delay'].mean(),
                    '平均帕累托点数': data['pareto_size'].mean(),
                    '平均运行时间': data['runtime'].mean(),
                }
                summary_data.append(row)
        
        # 计算改进率
        orig = df_valid[(df_valid['scale'] == scale) & (df_valid['version'] == 'original')]
        pls = df_valid[(df_valid['scale'] == scale) & (df_valid['version'] == 'with_pls')]
        if len(orig) > 0 and len(pls) > 0:
            orig_mean = orig['combined_objective'].mean()
            pls_mean = pls['combined_objective'].mean()
            if orig_mean > 0:
                improvement = (orig_mean - pls_mean) / orig_mean * 100
                print(f"\n📊 {scale}客户: PLS相比原始版本改进 {improvement:.2f}%")
    
    # 保存到CSV
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"comparison_table_{timestamp}.csv")
        df_summary.to_csv(save_path, index=False)
        print(f"\n✅ 对比表格已保存: {save_path}")


if __name__ == "__main__":
    # ===== 配置两个版本 =====
    versions = [
        ('with_pls', 'P-ACO_PLS'),        # 新版本文件名（不含.py）
        ('original', 'original_P-ACO'),   # 原始版本文件名（不含.py）
    ]
    
    # 创建结果目录
    os.makedirs("compare_results", exist_ok=True)
    for name, _ in versions:
        os.makedirs(f"compare_results/{name}", exist_ok=True)
    
    all_dfs = []
    
    for name, module in versions:
        print(f"\n{'='*60}")
        print(f"运行版本: {name}")
        print(f"{'='*60}")
        
        df = run_on_instances(name, module)
        all_dfs.append(df)
        
        # 保存单个版本结果
        if not df.empty:
            df.to_csv(f"compare_results/{name}/results.csv", index=False)
            print(f"✅ {name} 结果已保存到 compare_results/{name}/results.csv")
    
    # ===== 合并对比 =====
    valid_dfs = [df for df in all_dfs if not df.empty]
    if valid_dfs:
        combined = pd.concat(valid_dfs, ignore_index=True)
        
        # 保存汇总结果
        combined.to_csv("compare_results/all_results.csv", index=False)
        print("\n✅ 所有结果已保存到 compare_results/all_results.csv")
        
        # 打印对比表格
        print_comparison_table(combined, save_dir="compare_results")
        
        # 生成对比图
        generate_comparison_plots(combined, save_dir="compare_results")
        
    else:
        print("\n❌ 没有成功的结果可以汇总")