# 文件名: plot_group_distribution_figures.py (新脚本)
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ... (setup_matplotlib_for_publication 和 load_data 函数代码保持不变) ...
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
# 3. 新的绘图函数：绘制小组构成分布的演化
# =============================================================================
def plot_group_distribution_evolution(data_paths, output_filename, total_iterations=20000):
    """
    在一个Figure中为每个r值创建一个子图，
    每个子图展示拥有不同数量背叛者的小组占比的演化。
    """
    num_plots = len(data_paths)
    if num_plots == 0: return

    # --- 创建一个包含N行1列子图的Figure ---
    fig, axes = plt.subplots(num_plots, 1, 
                             figsize=(8, 5 * num_plots), 
                             sharex=True, sharey=True) # 共享X和Y轴
    
    if num_plots == 1:
        axes = [axes]

    # --- 定义6条曲线的样式 ---
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    
    # --- 遍历每个r值（每个子图） ---
    for i, (label, path) in enumerate(data_paths.items()):
        ax = axes[i]
        
        # 循环绘制6条曲线（0个背叛者到5个背叛者）
        for num_d in range(6):
            dataset_name = f"group_comp_d{num_d}_history"
            data = load_data(path, dataset_name)
            
            if data is not None:
                iterations = np.arange(1, len(data) + 1)
                ax.plot(iterations, data, lw=2, color=colors[num_d], 
                        label=f'{num_d} Defectors')
        
        # --- 格式化子图 ---
        ax.set_xscale('log')
        ax.set_xlim(1, total_iterations)
        ax.set_ylim(-5, 105) # Y轴为百分比
        ax.grid(True, which="both", ls="--", alpha=0.6)
        ax.set_ylabel('Percentage of Groups (%)')
        ax.set_title(label) # 例如: (a) r=1.0
        ax.legend(title="Group Type", loc='center left', bbox_to_anchor=(1, 0.5))

    axes[-1].set_xlabel('$t$ (Iteration)')

    # --- 保存和关闭 ---
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # 调整布局为图例留出空间
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Group distribution evolution figure saved to {output_filename}")


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
    r_values_to_compare = [1.0, 2.0, 3.0] 

    def get_folder_name(r, kappa, M_bool, wP, alpha, rep_gain_C):
        order_str = str(M_bool)
        return f"results_r{r:.1f}_inf{kappa:.1f}_order{order_str}_alpha{alpha}_rw{wP:.2f}_rgC{rep_gain_C:.2f}"

    comparison_paths = {
        f"({chr(ord('a') + i)}) $r={r_val:.1f}$": os.path.join(
            BASE_DATA_PATH, 
            get_folder_name(r=r_val, **FIXED_PARAMS), 
            "data/experiment_data.h5"
        )
        for i, r_val in enumerate(r_values_to_compare)
    }
    
    TOTAL_ITERATIONS = 20001

    print("\n--- Generating Group Distribution Evolution Figure ---")
    plot_group_distribution_evolution(
        comparison_paths, 
        os.path.join(OUTPUT_DIR, "Figure_Group_Distribution_Evolution.pdf"),
        total_iterations=TOTAL_ITERATIONS
    )
    print("\nPlotting task is complete.")