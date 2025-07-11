# 文件名: plot_combined_comparisons.py
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. 全局绘图与字体设置
# =============================================================================
def setup_matplotlib_for_publication(font_size_pt=12):
    plt.rcParams.update({
        "font.size": font_size_pt, "axes.titlesize": font_size_pt, "axes.labelsize": font_size_pt,
        "xtick.labelsize": font_size_pt - 2, "ytick.labelsize": font_size_pt - 2,
        "legend.fontsize": font_size_pt - 2, "figure.titlesize": font_size_pt + 2,
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "text.usetex": False, "figure.dpi": 300,
    })
    print(f"Matplotlib style updated for {font_size_pt}pt font.")

# =============================================================================
# 2. 数据加载函数
# =============================================================================
def load_data(filepath, dataset_name):
    if not os.path.exists(filepath):
        print(f"Warning: Data file not found at {filepath}")
        return None
    try:
        with h5py.File(filepath, 'r') as f:
            if dataset_name in f: return f[dataset_name][:]
            else:
                print(f"Warning: Dataset '{dataset_name}' not found in {filepath}")
                return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# =============================================================================
# 3. (新增) 合并绘图函数
# =============================================================================
def plot_combined_comparison(data_paths, output_filename, total_iterations=100001):
    """
    在一个Figure中绘制两个子图：左边是合作率对比，右边是策略转换数对比。
    """
    # --- 关键修改：创建一个包含1行2列子图的Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) # 调整figsize以容纳两个图

    # --- (a) 绘制左子图：合作率对比 ---
    for label, path in data_paths.items():
        coop_hist = load_data(path, 'coop_rate_history')
        if coop_hist is not None:
            plot_y = np.full(total_iterations, np.nan)
            plot_y[:len(coop_hist)] = coop_hist
            iterations = np.arange(1, total_iterations + 1)
            ax1.plot(iterations, plot_y, label=label, lw=1.5)

   
    ax1.set_xlabel('$t$ ')
    ax1.set_ylabel('$f_c$')
    ax1.set_xscale('log')
    ax1.set_title('(a)')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(1, total_iterations)
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.6)

    # --- (b) 绘制右子图：策略转换对比 ---
    for label, path in data_paths.items():
        switch_c_d = load_data(path, 'switch_C_to_D')
        switch_d_c = load_data(path, 'switch_D_to_C')
        if switch_c_d is not None and switch_d_c is not None:
            total_switches = switch_c_d + switch_d_c
            iterations = np.arange(1, len(total_switches) + 1)
            ax2.plot(iterations, total_switches, label=label, lw=1.5)
   
    
    ax2.set_xlabel('$t$ ')
    ax2.set_ylabel('Number of Strategy Switches')
    ax2.set_xscale('log')
    ax2.set_title('(b)')
    ax2.set_ylim(-5,10005) # 对数轴下限不能为0
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.6)

    # --- 统一调整布局并保存 ---
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Combined comparison figure saved to {output_filename}")


# =============================================================================
# 4. 主执行模块
# =============================================================================
if __name__ == '__main__':
    # --- 1. 设置绘图风格 ---
    setup_matplotlib_for_publication(font_size_pt=12)

    # --- 2. 指定数据根目录和输出目录 ---
    BASE_DATA_PATH = "./" 
    OUTPUT_DIR = "./paper_figures_final"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 3. 定义要对比的两个模型的数据路径 ---
    comparison_paths = {
        'State: Previous Action': os.path.join(BASE_DATA_PATH, "results_r4.6_inf0.0_orderFalse_alpha0.8_rw1.00_rgC1.00_action/data/experiment_data.h5"),
        'State: Reputation': os.path.join(BASE_DATA_PATH, "results_r4.6_inf0.0_orderFalse_alpha0.8_rw1.00_rgC1.00/data/experiment_data.h5"),
    }
    
    # 模拟的总迭代次数，用于统一x轴
    TOTAL_ITERATIONS = 100001
    
    # --- 4. 调用唯一的、合并的绘图函数 ---
    print("\n--- Generating Combined Comparison Figure ---")
    plot_combined_comparison(
        comparison_paths, 
        os.path.join(OUTPUT_DIR, "Figure_Coop_vs_Stability_Comparison.pdf"),
        total_iterations=TOTAL_ITERATIONS
    )
    
    print("\nAll plotting tasks are complete.")