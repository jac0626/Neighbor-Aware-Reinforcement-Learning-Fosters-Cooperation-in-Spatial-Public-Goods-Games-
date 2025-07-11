# 文件名: plot_all_figures_with_switches.py
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
# 3. 独立的绘图函数
# =============================================================================

# --- (在这里粘贴所有之前的 plot_figure_X 函数: plot_figure_2, 3, 4, 6, 7, 8, 9) ---
def plot_figure_2(data_paths, output_filename, total_iterations=200000):
    # ... (代码同上一版) ...
    pass
def plot_figure_3(data_paths, output_filename):
    # ... (代码同上一版) ...
    pass
# ... (其他绘图函数) ...

# --- (新增) 策略转换图绘图函数 ---
def plot_strategy_switches(data_paths, output_filename):
    """
    绘制策略转换数随时间的演化，以展示系统稳定性。
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))

    for label, path in data_paths.items():
        switch_c_d = load_data(path, 'switch_C_to_D')
        switch_d_c = load_data(path, 'switch_D_to_C')
        
        if switch_c_d is not None and switch_d_c is not None:
            total_switches = switch_c_d + switch_d_c
            # 归一化为总人口的百分比
            total_switch_rate_percent = total_switches / (100*100) * 100
            
            iterations = np.arange(1, len(total_switch_rate_percent) + 1)
            ax.plot(iterations, total_switch_rate_percent, label=label)

    ax.set_title('Evolution of Strategy Switching Rate')
    ax.set_xlabel('$t$ (Iteration)')
    ax.set_ylabel(r'Total Switch Rate (\%)')
    ax.set_xscale('log')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Strategy switch figure saved to {output_filename}")

# =============================================================================
# 4. 主执行模块
# =============================================================================
if __name__ == '__main__':
    # --- 1. 设置全局绘图风格 ---
    setup_matplotlib_for_publication(font_size_pt=12)

    # --- 2. 指定数据根目录和输出目录 ---
    BASE_DATA_PATH = "./" 
    OUTPUT_DIR = "./paper_figures_final"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # --- 3. 为每个图配置数据路径并调用绘图函数 ---
    
    def get_folder_name(r, kappa, M_bool, wP):
        order_str = 'True' if M_bool else 'False'
        return f"results_r{r:.1f}_inf{kappa:.1f}_order{order_str}_alpha0.8_rw{wP:.2f}_rgC1.00"

    # --- (在这里粘贴所有之前的绘图调用) ---
    # print("\n--- Generating Figure 2 ---")
    # ...
    # print("\n--- Generating Figure 9 ---")
    # ...

    # --- (新增) 绘制策略转换图 ---
    print("\n--- Generating Strategy Switch Figure (New Figure) ---")
    # 我们可以对比 Fig.2 中的几个关键kappa值
    # 例如，对比 kappa=0, 0.5, 2.0 在 M=1 的情况
    fig_switch_paths = {
        f"$\\kappa={k}$": os.path.join(get_folder_name(k, 0.0, False, 0.83), "data/experiment_data.h5")
        for k in [1.0,2.0,3.0]
    }
    plot_strategy_switches(fig_switch_paths, os.path.join(OUTPUT_DIR, "Figure_Strategy_Switches.pdf"))
    
    print("\nAll plotting tasks are complete.")