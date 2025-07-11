# 文件名: plot_q_value_figures.py
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
# 3. 新的Q值演化对比绘图函数 (核心功能)
# =============================================================================
def plot_q_evolution_comparison(data_paths, output_filename, total_iterations=20000):
    """
    在一个Figure中绘制多个并排的子图，每个子图展示一个参数设置下的Q值演化。
    """
    num_plots = len(data_paths)
    if num_plots == 0:
        print("No data paths provided for plotting.")
        return

    # --- 创建一个包含1行N列子图的Figure ---
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4.5), sharey=True)
    if num_plots == 1: # 确保axes总是一个可迭代的列表
        axes = [axes]

    # --- 定义Q值曲线的样式和标签 ---
    plot_styles = {
        'q_s0_c': {'color': 'blue',    'linestyle': '-',  'label': '$Q(s_0, C)$'},
        'q_s0_d': {'color': 'red',     'linestyle': '-',  'label': '$Q(s_0, D)$'},
        'q_s1_c': {'color': '#66c2a5', 'linestyle': '--', 'label': '$Q(s_1, C)$'}, # 使用更柔和的颜色
        'q_s1_d': {'color': '#fc8d62', 'linestyle': '--', 'label': '$Q(s_1, D)$'}  # 使用更柔和的颜色
    }
    q_keys_map = {
        'q_s0_c': 'avg_q_s0_c_history', 'q_s0_d': 'avg_q_s0_d_history',
        'q_s1_c': 'avg_q_s1_c_history', 'q_s1_d': 'avg_q_s1_d_history'
    }

    # --- 遍历每个参数设置并绘制子图 ---
    for i, (label, path) in enumerate(data_paths.items()):
        ax = axes[i]
        all_data_loaded = True
        
        # 加载并绘制4条Q值曲线
        for key, hdf5_key in q_keys_map.items():
            data = load_data(path, hdf5_key)
            if data is not None:
                # 统一X轴长度
                plot_y = np.full(total_iterations, np.nan)
                end_idx = min(len(data), total_iterations)
                plot_y[:end_idx] = data[:end_idx]
                iterations = np.arange(1, total_iterations + 1)
                ax.plot(iterations, plot_y, lw=1.5, **plot_styles[key])
            else:
                all_data_loaded = False
        
        # --- 子图格式化 ---
        if all_data_loaded:
            ax.set_xscale('log')
            ax.set_xlim(1, total_iterations)
            ax.grid(True, which="both", ls="--", alpha=0.6)
        else:
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', color='red')
            
        ax.set_title(label) # 使用字典的键作为子图标题，例如 "(a) r=1.0"
        ax.set_xlabel('$t$ (Iteration)')

    # --- 全局格式化 ---
    axes[0].set_ylabel('Average Q-value')

    # 创建一个共享图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4)

    plt.tight_layout(rect=[0, 0, 1, 0.92]) # 为图例留出空间
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Q-value comparison figure saved to {output_filename}")


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
    # 固定参数，用于定位文件夹
    FIXED_PARAMS = {
        'wP': 0.83,
        'kappa': 0.0,
        'M_bool': False,  # 假设 M=2, use_second_order=True
        'alpha': 0.8,
        'rep_gain_C': 1.0
    }
    
    # 变化的参数
    r_values_to_compare = [1.0, 2.0] 

    # 辅助函数，构造文件夹名称 (请确保与您的命名规则完全匹配)
    def get_folder_name(r, kappa, M_bool, wP, alpha, rep_gain_C):
        order_str = str(M_bool) # 'True' or 'False'
        return f"results_r{r:.1f}_inf{kappa:.1f}_order{order_str}_alpha{alpha}_rw{wP:.2f}_rgC{rep_gain_C:.2f}"

    # 构建数据路径字典
    # 字典的键将用作子图的标题
    comparison_paths = {
        f"(a) $r={r_val:.1f}$": os.path.join(
            BASE_DATA_PATH, 
            get_folder_name(r=r_val, **FIXED_PARAMS), 
            "data/experiment_data.h5"
        )
        for i, r_val in enumerate(r_values_to_compare)
                    
    }
    

    # 模拟的总迭代次数，用于统一x轴
    TOTAL_ITERATIONS = 10001

    # --- 4. 调用绘图函数 ---
    print("\n--- Generating Q-value Evolution Comparison Figure ---")
    for label, path in comparison_paths.items():
        print(f"  - Plotting for {label} from {path}")
        
    plot_q_evolution_comparison(
        comparison_paths, 
        os.path.join(OUTPUT_DIR, "Figure_Q_Value_Evolution.pdf") # 保存为PDF格式
    )
    
    print("\nQ-value plotting task is complete.")