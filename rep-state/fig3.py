import os
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# =============================================================================
# 独立的、自包含的脚本，专门用于绘制Figure 3
# =============================================================================

def setup_matplotlib_for_publication(font_size_pt=12):
    """
    为出版物设置统一的Matplotlib风格。
    """
    # 强制使用非交互式后端，确保兼容性
    mpl.use('Agg')
    
    plt.rcParams.update({
        "font.size": font_size_pt,
        "axes.titlesize": font_size_pt,
        "axes.labelsize": font_size_pt,
        "xtick.labelsize": font_size_pt - 2,
        "ytick.labelsize": font_size_pt - 2,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "text.usetex": False, 
        "figure.dpi": 300,
    })
    print(f"Matplotlib style updated for {font_size_pt}pt font.")

def load_snapshot_data(filepath, time_point):
    """
    从指定的HDF5文件中加载特定的策略快照数据集。
    """
    dataset_name = f"Sn_snapshot_{time_point}"
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}")
        return None
    try:
        with h5py.File(filepath, 'r') as f:
            if dataset_name in f:
                snapshot = f[dataset_name][:]
                # 调试信息：检查数据内容
                print(f"Loaded snapshot for t={time_point}: shape={snapshot.shape}, min={np.min(snapshot)}, max={np.max(snapshot)}, mean={np.mean(snapshot)}")
                unique, counts = np.unique(snapshot, return_counts=True)
                print(f"Unique values: {unique}, Counts: {counts}")
                return snapshot
            else:
                print(f"Error: Dataset '{dataset_name}' not found in {filepath}")
                return None
    except Exception as e:
        print(f"Error loading snapshot from {filepath}: {e}")
        return None

def main():
    """
    主执行函数：加载数据并绘制Figure 3。
    """
    # --- 1. 设置绘图风格 ---
    setup_matplotlib_for_publication(font_size_pt=12)

    # --- 2. 配置项：指定数据文件路径 ---
    dir_k0 = "results_r3.6_inf0.0_orderFalse_alpha0.8_rw1.00_rgC1.00"
    dir_k0_5 = "results_r3.6_inf0.5_orderFalse_alpha0.8_rw1.00_rgC1.00"
    
    data_paths = {
        '$\\kappa=0$': os.path.join(dir_k0, "data", "experiment_data.h5"),
        '$\\kappa=0.5$': os.path.join(dir_k0_5, "data", "experiment_data.h5")
    }
    
    # --- 3. 指定输出文件名 ---
    OUTPUT_FILENAME = "Figure_3_standalone.pdf"

    # --- 4. 开始绘图 ---
    time_points = [100, 1000, 10000]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7, 4.8), gridspec_kw={'hspace': 0.3, 'wspace': 0.1})
    
    # 定义RGB颜色 (0-1范围)
    color_coop = np.array([192, 192, 192]) / 255.0  # 浅灰色 (合作者, 值为0)
    color_defect = np.array([0, 0, 0]) / 255.0      # 黑色 (背叛者, 值为1)

    for i, (label, path) in enumerate(data_paths.items()):
        axes[i, 0].set_ylabel(label, fontsize=plt.rcParams['axes.labelsize'], rotation=90, labelpad=20)
        
        for j, t in enumerate(time_points):
            ax = axes[i, j]
            snapshot = load_snapshot_data(path, t)
            
            if snapshot is not None:
                # 初始化为红色，便于调试异常值
                rgb_image = np.full((snapshot.shape[0], snapshot.shape[1], 3), [1, 0, 0])
                rgb_image[snapshot == 0] = color_coop
                rgb_image[snapshot == 1] = color_defect
                
                # 调试：检查RGB图像的像素值
                print(f"rgb_image[0,0] for {label}, t={t}: {rgb_image[0,0]}")
                print(f"rgb_image[-1,-1] for {label}, t={t}: {rgb_image[-1,-1]}")
                
                ax.imshow(rgb_image, interpolation='nearest', aspect='equal')
            else:
                ax.text(0.5, 0.5, 'Data\nMissing', ha='center', va='center', transform=ax.transAxes, fontsize=8, color='red')
            
            # 设置顶部标题
            if i == 0:
                ax.set_title(f"$t={t}$")
            
            # 隐藏坐标轴
            ax.set_xticks([])
            ax.set_yticks([])

    # 保存图片
    plt.savefig(OUTPUT_FILENAME, bbox_inches='tight')
    plt.savefig("Figure_3_standalone.png", bbox_inches='tight')  # 额外保存PNG以便检查
    plt.close(fig)
    print(f"\nFigure 3 has been generated and saved as: {OUTPUT_FILENAME} and Figure_3_standalone.png")

if __name__ == '__main__':
    main()