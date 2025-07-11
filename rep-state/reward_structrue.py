# 文件名: plot_reward_and_coop_side_by_side.py
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
# 3. (修改后) 合并绘图函数
# =============================================================================
def plot_reward_and_coop_side_by_side(data_paths, output_filename, total_iterations=100000):
    """
    在一个Figure中绘制两个左右并排的子图：
    左图：不同r值下的合作率演化对比。
    右图：不同r值下的声誉奖励占比演化对比。
    """
    # --- 关键修改：创建一个包含1行2列子图的Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True) # 共享X轴

    # 遍历每个r值对应的数据
    for label, path in data_paths.items():
        # --- (a) 绘制左子图：合作率对比 ---
        coop_hist = load_data(path, 'coop_rate_history')
        if coop_hist is not None:
            # 确保X轴长度一致
            plot_y = np.full(total_iterations, np.nan)
            plot_y[:total_iterations] = coop_hist[:total_iterations]
            iterations = np.arange(1, total_iterations + 1)
            ax1.plot(iterations, plot_y, label=label, lw=1.5)

        # --- (b) 绘制右子图：声誉奖励占比对比 ---
        ratio_hist = load_data(path, 'reputation_reward_ratio')
        if ratio_hist is not None:
            iterations = np.arange(1, total_iterations + 1)
            ax2.plot(iterations, ratio_hist[:total_iterations], label=label, lw=1.5)

    # --- 统一设置左子图 (合作率) 的格式 ---
    ax1.set_title('(a) Cooperation Dynamics')
    ax1.set_xlabel('$t$ ')
    ax1.set_ylabel('Fraction of Cooperation ($f_c$)')
    ax1.set_xscale('log')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(1, total_iterations)
    ax1.legend(title="$r$ value")
    ax1.grid(True, which="both", ls="--", alpha=0.6)

    # --- 统一设置右子图 (声誉占比) 的格式 ---
    ax2.set_title('(b) Reward Structure')
    ax2.set_xlabel('$t$ ')
    ax2.set_xscale('log')
    ax2.set_ylabel(r'Reputation Reward Ratio (\%)')
    # ax2.set_ylim(bottom=0) # Y轴范围可以根据数据动态调整
    ax2.legend(title="$r$ value")
    ax2.grid(True, which="both", ls="--", alpha=0.6)

    # --- 统一调整布局并保存 ---
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Side-by-side comparison figure saved to {output_filename}")


# =============================================================================
# 4. 主执行模块
# =============================================================================
if __name__ == '__main__':
    # --- 1. 设置绘图风格 ---
    setup_matplotlib_for_publication(font_size_pt=12)

    # --- 2. 指定数据根目录和输出目录 ---
    BASE_DATA_PATH = "./" 
    OUTPUT_DIR = "./paper_figures_final"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # --- 3. 定义要对比的参数组合 ---
    # 这是审稿人指出反常现象的区域
    FIXED_WP = 0.83
    # 选择几个r值进行对比，以展示趋势
    r_values_to_compare = [1.0,2.0] 

    # 辅助函数，构造文件夹名称
    def get_folder_name(r, kappa, M_bool, wP):
        order_str = 'True' if M_bool else 'False'
        # !! 关键 !!: 确保这里的命名规则与您的文件夹完全匹配
        return f"results_r{r:.1f}_inf{kappa:.1f}_order{order_str}_alpha0.8_rw{wP:.2f}_rgC1.00"

    # 构建数据路径字典
    comparison_paths = {
        f"$r={r_val:.1f}$": os.path.join(
            BASE_DATA_PATH, 
            get_folder_name(r=r_val, kappa=0.0, M_bool=False, wP=FIXED_WP), 
            "data/experiment_data.h5"
        )
        for r_val in r_values_to_compare
    }
    
    # 模拟的总迭代次数，用于统一x轴
    TOTAL_ITERATIONS = 20001

    # --- 4. 调用绘图函数 ---
    print("\n--- Generating Side-by-Side Comparison Figure ---")
    plot_reward_and_coop_side_by_side(
        comparison_paths, 
        os.path.join(OUTPUT_DIR, "Figure_Reward_vs_Coop_SideBySide.pdf"),
        total_iterations=TOTAL_ITERATIONS
    )
    
    print("\nAll plotting tasks are complete.")