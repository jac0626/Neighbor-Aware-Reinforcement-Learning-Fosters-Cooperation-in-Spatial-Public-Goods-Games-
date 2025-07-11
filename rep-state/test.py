import numpy as np
import matplotlib.pyplot as plt

# --- 1. 定义理论计算函数 ---

def calculate_payoff_advantage(r, w_p):
    """
    计算背叛者的加权归一化收益优势
    公式: w_p * (P'_D(r) - P'_C(r)) = w_p * (5 - r) / (3r + 5)
    
    Args:
        r (np.array or float): 协同因子
        w_p (float): 收益权重
        
    Returns:
        np.array or float: 收益优势值
    """
    # 避免r为负数或导致分母为零的情况
    r = np.asarray(r)
    # 确保 r > -5/3 to avoid division by zero or negative denominator
    if np.any(r <= -5/3):
        raise ValueError("Synergy factor 'r' must be greater than -5/3.")
        
    advantage = w_p * (5 - r) / (3 * r + 5)
    return advantage

def calculate_reputation_subsidy(w_p):
    """
    计算合作者的声誉补贴
    公式: 1 - w_p
    
    Args:
        w_p (float): 收益权重
        
    Returns:
        float: 声誉补贴值
    """
    return 1 - w_p

# --- 2. 绘图设置 ---

# 设置全局字体和样式，使其更适合论文
plt.style.use('seaborn-v0_8-whitegrid') # 使用一个干净的网格样式
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'font.family': 'serif', # 使用衬线字体，更正式
    'mathtext.fontset': 'dejavuserif'
})

# 创建一个Figure和两个子图(ax1, ax2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.suptitle('Theoretical Analysis of Reward Competition', fontsize=16, y=1.02)

# 定义r的范围
r_values = np.linspace(1.0, 5.0, 400) # 从1到5生成400个点，使曲线平滑

# --- 3. 绘制第一个子图 (a) Critical Case: w_p = 0.8 ---

w_p_critical = 0.83
payoff_advantage_critical = calculate_payoff_advantage(r_values, w_p_critical)
reputation_subsidy_critical = calculate_reputation_subsidy(w_p_critical)

# 绘制曲线
ax1.plot(r_values, payoff_advantage_critical, label="Defector's Payoff Advantage", color='orangered', linewidth=2.5)
ax1.axhline(y=reputation_subsidy_critical, label="Cooperator's Reputation Subsidy", color='dodgerblue', linestyle='--', linewidth=2.5)

# 找到交叉点
intersection_idx = np.argwhere(np.diff(np.sign(payoff_advantage_critical - reputation_subsidy_critical))).flatten()
if intersection_idx.size > 0:
    r_intersect = r_values[intersection_idx[0]]
    y_intersect = reputation_subsidy_critical
    ax1.plot(r_intersect, y_intersect, 'ko', markersize=8, label=f'Intersection ($r \\approx {r_intersect:.2f}$)')
    # 添加垂直线指示交叉点
    ax1.vlines(r_intersect, 0, y_intersect, colors='gray', linestyles='dotted', linewidth=1.5)

# 区域填充，增加可视化效果
ax1.fill_between(r_values, payoff_advantage_critical, reputation_subsidy_critical, 
                 where=(reputation_subsidy_critical > payoff_advantage_critical), 
                 color='dodgerblue', alpha=0.2, label='Cooperation Favored')
ax1.fill_between(r_values, payoff_advantage_critical, reputation_subsidy_critical, 
                 where=(payoff_advantage_critical > reputation_subsidy_critical), 
                 color='orangered', alpha=0.2, label='Defection Favored')


# 设置子图标题和标签
ax1.set_title(f'(a) Critical Case: $w_P = {w_p_critical}$', fontsize=14)
ax1.set_xlabel('Synergy Factor ($r$)')
ax1.set_ylabel('Reward Component Value')
ax1.legend(loc='upper right')
ax1.set_xlim(1, 5)
ax1.set_ylim(0, 0.4)


# --- 4. 绘制第二个子图 (b) Normal Case: w_p = 0.7 ---

w_p_normal = 0.7
payoff_advantage_normal = calculate_payoff_advantage(r_values, w_p_normal)
reputation_subsidy_normal = calculate_reputation_subsidy(w_p_normal)

# 绘制曲线
ax2.plot(r_values, payoff_advantage_normal, label="Defector's Payoff Advantage", color='orangered', linewidth=2.5)
ax2.axhline(y=reputation_subsidy_normal, label="Cooperator's Reputation Subsidy", color='dodgerblue', linestyle='--', linewidth=2.5)

# 区域填充
ax2.fill_between(r_values, payoff_advantage_normal, reputation_subsidy_normal, 
                 where=(reputation_subsidy_normal > payoff_advantage_normal), 
                 color='dodgerblue', alpha=0.2, label='Cooperation Favored')


# 设置子图标题和标签
ax2.set_title(f'(b) Normal Case: $w_P = {w_p_normal}$', fontsize=14)
ax2.set_xlabel('Synergy Factor ($r$)')
# ax2.set_ylabel('Reward Component Value') # Y轴共享，无需重复设置
ax2.legend(loc='upper right')
ax2.set_xlim(1, 5)

# --- 5. 调整布局并保存图像 ---

plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局，防止标题重叠
plt.savefig('reward_competition_analysis.png', dpi=300, bbox_inches='tight')
plt.show()