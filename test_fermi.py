#!/usr/bin/env python3
"""
经典二策略Fermi SPGG - 基于notebook原始实现
测试临界点

用法:
    python test_fermi.py --r 3.6 --L 100 --iterations 5000
    python test_fermi.py --sweep  # 扫描多个r值
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 计算5邻域重叠
overlap5 = lambda A: A + np.roll(A, -1, 0) + np.roll(A, 1, 0) + np.roll(A, -1, 1) + np.roll(A, 1, 1)

class FermiSPGG:
    """
    经典空间公共物品博弈 + Fermi更新规则
    
    二策略版本:
    - 策略0: 合作者 (Cooperator)
    - 策略1: 背叛者 (Defector)
    
    Payoff计算 (每个5人组):
    - 合作者: r * c * Nc / 5 - cost
    - 背叛者: r * c * Nc / 5
    
    每个agent参与5个组，总payoff = 5个组的payoff之和
    """
    
    def __init__(self, r=3.0, c=1.0, cost=1.0, K=0.1, L=100, iterations=5000, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.r = r
        self.c = c
        self.cost = cost
        self.K = K
        self.L = L
        self.iterations = iterations
        
        # 初始化: 随机50%合作者、50%背叛者
        # 0 = 合作者, 1 = 背叛者
        self._Sn = np.random.randint(0, 2, size=(L, L))
        self._update_S()
        
    def _update_S(self):
        """更新策略矩阵"""
        self._S = [(self._Sn == j).astype(int) for j in range(2)]
    
    def N(self, group_offset=(0, 0)):
        """计算每个组中各策略的数量"""
        S = self._S
        if group_offset != (0, 0):
            S = [np.roll(s, group_offset, axis=(0, 1)) for s in S]
        return [overlap5(s) for s in S]
    
    def P_g_m(self, group_offset=(0, 0), member_offset=(0, 0)):
        """
        计算单个组的payoff
        
        group_offset: 组的中心相对于当前位置的偏移
        member_offset: 成员位置的偏移
        """
        N = self.N(group_offset)
        S = self._S
        if group_offset != (0, 0):
            S = [np.roll(s, group_offset, axis=(0, 1)) for s in S]
        if member_offset != (0, 0):
            S = [np.roll(s, member_offset, axis=(0, 1)) for s in S]
        
        r, c, cost = self.r, self.c, self.cost
        n = 5
        Nc = N[0]  # 合作者数量
        S_coop, S_defect = S[0], S[1]
        
        # Payoff:
        # 合作者: r*c*Nc/n - cost
        # 背叛者: r*c*Nc/n
        P = (r * c * Nc / n - cost) * S_coop + (r * c * Nc / n) * S_defect
        return P
    
    def compute_total_payoff(self):
        """计算总payoff (5个组的累计)"""
        # 注意: 原始notebook的offset对应关系
        P = (self.P_g_m() + 
             self.P_g_m((1, 0), (-1, 0)) + 
             self.P_g_m((-1, 0), (1, 0)) + 
             self.P_g_m((0, 1), (0, -1)) + 
             self.P_g_m((0, -1), (0, 1)))
        return P
    
    def fermi_update(self, P):
        """
        Fermi更新规则
        
        每个agent随机选择一个邻居，以Fermi概率学习其策略
        W = 1 / (1 + exp((P_self - P_neighbor) / K))
        """
        L, K = self.L, self.K
        S_in_one = self._Sn
        
        # 计算四个方向的Fermi概率
        W_w = 1 / (1 + np.exp((P - np.roll(P, 1, 1)) / K))   # 西
        W_e = 1 / (1 + np.exp((P - np.roll(P, -1, 1)) / K))  # 东  
        W_n = 1 / (1 + np.exp((P - np.roll(P, 1, 0)) / K))   # 北
        W_s = 1 / (1 + np.exp((P - np.roll(P, -1, 0)) / K))  # 南
        
        # 随机选择邻居 (0=西, 1=东, 2=北, 3=南)
        RandomNeighbour = np.random.randint(0, 4, size=(L, L))
        Random01 = np.random.uniform(0, 1, size=(L, L))
        
        # 根据Fermi概率决定是否学习邻居的策略
        S_new = ((RandomNeighbour == 0) * ((Random01 <= W_w) * np.roll(S_in_one, 1, 1) + (Random01 > W_w) * S_in_one) +
                 (RandomNeighbour == 1) * ((Random01 <= W_e) * np.roll(S_in_one, -1, 1) + (Random01 > W_e) * S_in_one) +
                 (RandomNeighbour == 2) * ((Random01 <= W_n) * np.roll(S_in_one, 1, 0) + (Random01 > W_n) * S_in_one) +
                 (RandomNeighbour == 3) * ((Random01 <= W_s) * np.roll(S_in_one, -1, 0) + (Random01 > W_s) * S_in_one))
        
        self._Sn = S_new.astype(int)
        self._update_S()
    
    def run(self, log=False):
        """运行模拟"""
        coop_history = []
        
        for i in range(self.iterations):
            # 记录合作率
            coop_rate = np.sum(self._S[0]) / (self.L * self.L)
            coop_history.append(coop_rate)
            
            if log and i % 500 == 0:
                print(f"  Iteration {i}: coop_rate = {coop_rate:.4f}")
            
            # 计算payoff
            P = self.compute_total_payoff()
            
            # Fermi更新
            self.fermi_update(P)
            
            # 早停条件
            if coop_rate <= 0.001 or coop_rate >= 0.999:
                coop_history.extend([coop_rate] * (self.iterations - i - 1))
                break
        
        return np.array(coop_history)


def test_single_r(r, L=100, iterations=5000, K=0.1, seeds=[42, 123, 456]):
    """测试单个r值"""
    print(f"\nTesting r = {r}:")
    results = []
    for seed in seeds:
        model = FermiSPGG(r=r, L=L, iterations=iterations, K=K, seed=seed)
        history = model.run()
        results.append(history[-1])
        print(f"  Seed {seed}: final coop = {history[-1]:.4f}")
    print(f"  Mean: {np.mean(results):.4f}")
    return results


def sweep_r_values(r_values, L=100, iterations=5000, K=0.1, n_runs=3):
    """扫描多个r值"""
    print(f"Sweeping r values with L={L}, iterations={iterations}, K={K}")
    print("=" * 60)
    
    all_results = {}
    for r in r_values:
        results = []
        for run in range(n_runs):
            model = FermiSPGG(r=r, L=L, iterations=iterations, K=K, seed=run*100+42)
            history = model.run()
            results.append(history[-1])
        all_results[r] = results
        mean = np.mean(results)
        print(f"r = {r:.2f}: mean = {mean:.4f}, runs = {[f'{x:.3f}' for x in results]}")
    
    # 绘图
    r_list = list(all_results.keys())
    means = [np.mean(all_results[r]) for r in r_list]
    stds = [np.std(all_results[r]) for r in r_list]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(r_list, means, yerr=stds, fmt='o-', capsize=5)
    plt.axhline(y=0.5, color='r', linestyle='--', label='50% cooperation')
    plt.xlabel('Synergy factor r')
    plt.ylabel('Final cooperation rate')
    plt.title(f'Classic Fermi SPGG (L={L}, K={K}, iterations={iterations})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fermi_r_sweep.png', dpi=150)
    plt.show()
    print("\nPlot saved to fermi_r_sweep.png")
    
    return all_results


def plot_evolution(r_values, L=100, iterations=5000, K=0.1, seed=42):
    """绘制演化曲线"""
    plt.figure(figsize=(10, 6))
    
    for r in r_values:
        model = FermiSPGG(r=r, L=L, iterations=iterations, K=K, seed=seed)
        history = model.run()
        plt.semilogx(history, label=f'r={r}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Cooperation rate')
    plt.title(f'Evolution of cooperation (L={L}, K={K})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.savefig('fermi_evolution.png', dpi=150)
    plt.show()
    print("\nPlot saved to fermi_evolution.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test classic Fermi SPGG')
    parser.add_argument('--r', type=float, default=3.6, help='Synergy factor')
    parser.add_argument('--L', type=int, default=100, help='Grid size')
    parser.add_argument('--iterations', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--K', type=float, default=0.1, help='Temperature (selection strength)')
    parser.add_argument('--sweep', action='store_true', help='Sweep multiple r values')
    parser.add_argument('--evolution', action='store_true', help='Plot evolution curves')
    args = parser.parse_args()
    
    if args.sweep:
        r_values = [2.0, 2.5, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0]
        sweep_r_values(r_values, L=args.L, iterations=args.iterations, K=args.K)
    elif args.evolution:
        r_values = [2.5, 3.0, 3.5, 4.0, 4.5]
        plot_evolution(r_values, L=args.L, iterations=args.iterations, K=args.K)
    else:
        test_single_r(args.r, L=args.L, iterations=args.iterations, K=args.K)
