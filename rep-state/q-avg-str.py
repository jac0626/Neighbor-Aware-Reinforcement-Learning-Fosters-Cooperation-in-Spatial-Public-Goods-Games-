# 文件名: plot_q_value_figures.py
# (保留 setup_matplotlib_for_publication 和 load_data 函数)
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. 全局绘图与字体设置 (与您的规范完全一致)
# =============================================================================
def setup_matplotlib_for_publication(font_size_pt=12):
    """
    配置Matplotlib以生成出版物质量的图表。
    """
    plt.rcParams.update({
        "font.size": font_size_pt, "axes.titlesize": font_size_pt, "axes.labelsize": font_size_pt,
        "xtick.labelsize": font_size_pt - 2, "ytick.labelsize": font_size_pt - 2,
        "legend.fontsize": font_size_pt - 2, "figure.titlesize": font_size_pt + 2,
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "text.usetex": False, # 如果需要LaTeX公式，可以设为True
        "figure.dpi": 300,
    })
    print(f"Matplotlib style updated for {font_size_pt}pt font.")

# =============================================================================
# 2. 数据加载函数 (与您的规范完全一致)
# =============================================================================
def load_data(filepath, dataset_name):
    """
    从HDF5文件中安全地加载指定的数据集。
    """
    if not os.path.exists(filepath):
        print(f"Warning: Data file not found at {filepath}")
        return None
    try:
        with h5py.File(filepath, 'r') as f:
            if dataset_name in f:
                return f[dataset_name][:]
            else:
                print(f"Warning: Dataset '{dataset_name}' not found in {filepath}")
                return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# =============================================================================
# 3. 新的、按策略群体划分的Q值演化绘图函数
# =============================================================================
def plot_q_by_strategy_comparison(data_paths, output_filename, total_iterations=20000):
    """
    在一个Figure中绘制多个面板，每个面板包含左右两个子图，
    分别展示合作者和背叛者群体的Q值演化。
    """
    num_param_sets = len(data_paths)
    if num_param_sets == 0: return

    # --- 每个参数设置一行，每行包含两个子图（合作者/背叛者） ---
    fig, axes = plt.subplots(num_param_sets, 2, 
                             figsize=(10, 4.5 * num_param_sets), 
                             sharey=True) # 所有子图共享Y轴
    
    if num_param_sets == 1: # 确保axes是二维数组
        axes = np.array([axes])

    # --- 定义样式和标签 ---
    plot_styles = {
        'q_s0_c': {'color': 'blue',    'linestyle': '-',  'label': '$Q(s_0, C)$'},
        'q_s0_d': {'color': 'red',     'linestyle': '-',  'label': '$Q(s_0, D)$'},
        'q_s1_c': {'color': '#66c2a5', 'linestyle': '--', 'label': '$Q(s_1, C)$'},
        'q_s1_d': {'color': '#fc8d62', 'linestyle': '--', 'label': '$Q(s_1, D)$'}
    }
    q_keys_map = {key: f"_{key}_history" for key in plot_styles}

    # --- 遍历每个参数设置（对应每一行） ---
    for i, (row_label, path) in enumerate(data_paths.items()):
        ax_coop = axes[i, 0]
        ax_def = axes[i, 1]

        # --- (a) 绘制左侧子图：合作者群体的Q值 ---
        for key, hdf5_suffix in q_keys_map.items():
            dataset_name = f"cooperators{hdf5_suffix}"
            data = load_data(path, dataset_name)
            if data is not None:
                iterations = np.arange(1, len(data) + 1)
                ax_coop.plot(iterations, data, lw=1.5, **plot_styles[key])
        
        # --- (b) 绘制右侧子图：背叛者群体的Q值 ---
        for key, hdf5_suffix in q_keys_map.items():
            dataset_name = f"defectors{hdf5_suffix}"
            data = load_data(path, dataset_name)
            if data is not None:
                iterations = np.arange(1, len(data) + 1)
                ax_def.plot(iterations, data, lw=1.5, **plot_styles[key])

        # --- 格式化子图 ---
        for ax in [ax_coop, ax_def]:
            ax.set_xscale('log')
            ax.set_xlim(1, total_iterations)
            ax.grid(True, which="both", ls="--", alpha=0.6)
            if i == num_param_sets - 1: # 只在最底部的子图标注x轴
                ax.set_xlabel('$t$ (Iteration)')
        
        ax_coop.set_ylabel('Average Q-value')
        
        # 设置行标题（在左侧子图的ylabel位置）
        ax_coop.set_ylabel(f"{row_label}\n\nAverage Q-value", multialignment='center')


    # 设置列标题
    axes[0, 0].set_title('Q-values of Cooperators')
    axes[0, 1].set_title('Q-values of Defectors')

    # --- 全局图例和布局 ---
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4)
    plt.tight_layout(rect=[0, 0, 1, 0.94]) # 调整布局为图例留出空间
    
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Q-value by strategy figure saved to {output_filename}")


# =============================================================================
# 4. 主执行模块
# =============================================================================
if __name__ == '__main__':
    setup_matplotlib_for_publication(font_size_pt=12)
    BASE_DATA_PATH = "./" 
    OUTPUT_DIR = "./paper_figures_final"
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    FIXED_PARAMS = {
        'wP': 0.83, 'kappa': 0.0, 'M_bool': False,
        'alpha': 0.8, 'rep_gain_C': 1.0
    }
    r_values_to_compare = [1.0, 2.0,3.0] 

    def get_folder_name(r, kappa, M_bool, wP, alpha, rep_gain_C):
        order_str = str(M_bool)
        return f"results_r{r:.1f}_inf{kappa:.1f}_order{order_str}_alpha{alpha}_rw{wP:.2f}_rgC{rep_gain_C:.2f}"

    comparison_paths = {
        f"$r={r_val:.1f}$": os.path.join(
            BASE_DATA_PATH, 
            get_folder_name(r=r_val, **FIXED_PARAMS), 
            "data/experiment_data.h5"
        )
        for r_val in r_values_to_compare
    }

    TOTAL_ITERATIONS = 20001

    print("\n--- Generating Q-value by Strategy Comparison Figure ---")
    plot_q_by_strategy_comparison(
        comparison_paths, 
        os.path.join(OUTPUT_DIR, "Figure_Q_Value_by_Strategy.pdf"),
        total_iterations=TOTAL_ITERATIONS
    )
    print("\nPlotting task is complete.")