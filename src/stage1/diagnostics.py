"""
因果推論診断可視化ツール
NASA JPL標準：すべての判断に視覚的証拠を

Google/Meta実例：A/Bテストレポートで必須の診断図
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler

# スタイル設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class CausalDiagnostics:
    """
    因果推論の診断と可視化
    
    主要機能：
    1. Love Plot（共変量バランス）
    2. Propensity Score分布
    3. Overlap診断
    4. 統合診断レポート
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.diagnostics_history = []
        
    def create_love_plot(
        self,
        X: np.ndarray,
        T: np.ndarray,
        weights: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        Love Plot: 共変量バランスの可視化
        
        SMD (Standardized Mean Difference) < 0.1 が目標
        """
        n_features = X.shape[1]
        
        if feature_names is None:
            feature_names = [f"X{i+1}" for i in range(n_features)]
            
        # SMD計算（重み付き・なし両方）
        smd_unweighted = []
        smd_weighted = []
        
        for i in range(n_features):
            # Unweighted SMD
            smd_uw = self._calculate_smd(X[:, i], T, None)
            smd_unweighted.append(smd_uw)
            
            # Weighted SMD
            if weights is not None:
                smd_w = self._calculate_smd(X[:, i], T, weights)
                smd_weighted.append(smd_w)
            else:
                smd_weighted.append(smd_uw)
        
        # プロット作成
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        y_pos = np.arange(len(feature_names))
        
        # Unweighted (before)
        ax.scatter(smd_unweighted, y_pos, 
                  s=100, alpha=0.6, label='Before weighting',
                  color='coral', marker='o')
        
        # Weighted (after)
        if weights is not None:
            ax.scatter(smd_weighted, y_pos,
                      s=100, alpha=0.8, label='After weighting',
                      color='steelblue', marker='s')
            
            # 改善を示す矢印
            for i in range(len(feature_names)):
                ax.annotate('', xy=(smd_weighted[i], i),
                           xytext=(smd_unweighted[i], i),
                           arrowprops=dict(arrowstyle='->', alpha=0.3, color='gray'))
        
        # 閾値ライン
        ax.axvline(x=-threshold, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=threshold, color='red', linestyle='--', alpha=0.5)
        ax.axvspan(-threshold, threshold, alpha=0.1, color='green')
        
        # ラベルと装飾
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Standardized Mean Difference (SMD)', fontsize=12)
        ax.set_title('Love Plot: Covariate Balance Assessment', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Balance statistics
        balance_stats = {
            'n_balanced': sum(abs(s) < threshold for s in smd_weighted),
            'n_total': n_features,
            'max_smd': max(abs(s) for s in smd_weighted),
            'mean_smd': np.mean([abs(s) for s in smd_weighted])
        }
        
        # テキストボックスで統計表示
        textstr = f'Balanced: {balance_stats["n_balanced"]}/{balance_stats["n_total"]}\n'
        textstr += f'Max SMD: {balance_stats["max_smd"]:.3f}\n'
        textstr += f'Mean SMD: {balance_stats["mean_smd"]:.3f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show()
        
        return balance_stats
    
    def plot_propensity_score_distribution(
        self,
        ps: np.ndarray,
        T: np.ndarray,
        trim_threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Propensity Score分布の可視化
        
        Positivity違反の検出に重要
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. Histogram by treatment group
        ax = axes[0, 0]
        ax.hist(ps[T == 1], bins=30, alpha=0.5, label='Treated', color='coral', density=True)
        ax.hist(ps[T == 0], bins=30, alpha=0.5, label='Control', color='steelblue', density=True)
        
        if trim_threshold:
            ax.axvline(x=trim_threshold, color='red', linestyle='--', alpha=0.7)
            ax.axvline(x=1-trim_threshold, color='red', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Propensity Score')
        ax.set_ylabel('Density')
        ax.set_title('PS Distribution by Treatment Group')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Overlap region (mirror histogram)
        ax = axes[0, 1]
        bins = np.linspace(0, 1, 31)
        
        # Treated (上向き)
        counts_t, _ = np.histogram(ps[T == 1], bins=bins)
        # Control (下向き)
        counts_c, _ = np.histogram(ps[T == 0], bins=bins)
        
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2
        
        ax.bar(center, counts_t, width=width, alpha=0.5, label='Treated', color='coral')
        ax.bar(center, -counts_c, width=width, alpha=0.5, label='Control', color='steelblue')
        
        ax.set_xlabel('Propensity Score')
        ax.set_ylabel('Count (Treated ↑ / Control ↓)')
        ax.set_title('Mirror Histogram: Overlap Assessment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        # 3. Logit PS distribution（より診断的）
        ax = axes[1, 0]
        logit_ps = np.log(ps / (1 - ps + 1e-10))
        
        ax.hist(logit_ps[T == 1], bins=30, alpha=0.5, label='Treated', 
                color='coral', density=True)
        ax.hist(logit_ps[T == 0], bins=30, alpha=0.5, label='Control',
                color='steelblue', density=True)
        
        ax.set_xlabel('Logit(Propensity Score)')
        ax.set_ylabel('Density')
        ax.set_title('Logit-scale PS Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Density overlap plot
        ax = axes[1, 1]
        from scipy.stats import gaussian_kde
        
        # KDE推定
        kde_t = gaussian_kde(ps[T == 1])
        kde_c = gaussian_kde(ps[T == 0])
        
        ps_range = np.linspace(0, 1, 200)
        density_t = kde_t(ps_range)
        density_c = kde_c(ps_range)
        
        # Overlap領域の計算
        overlap = np.minimum(density_t, density_c)
        
        ax.fill_between(ps_range, 0, density_t, alpha=0.3, color='coral', label='Treated')
        ax.fill_between(ps_range, 0, density_c, alpha=0.3, color='steelblue', label='Control')
        ax.fill_between(ps_range, 0, overlap, alpha=0.5, color='purple', label='Overlap')
        
        ax.set_xlabel('Propensity Score')
        ax.set_ylabel('Density')
        ax.set_title('Density Overlap')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Overlap統計
        overlap_area = np.trapz(overlap, ps_range)
        total_area = np.trapz(density_t, ps_range) + np.trapz(density_c, ps_range)
        overlap_ratio = 2 * overlap_area / total_area  # 正規化
        
        # 統計サマリー
        stats = {
            'overlap_ratio': overlap_ratio,
            'n_extreme_low': np.sum(ps < 0.01),
            'n_extreme_high': np.sum(ps > 0.99),
            'min_ps': np.min(ps),
            'max_ps': np.max(ps)
        }
        
        plt.suptitle(f'Propensity Score Diagnostics (Overlap: {overlap_ratio:.1%})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return stats
    
    def overlap_diagnostic_plot(
        self,
        ps: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> None:
        """
        Overlap診断の統合プロット
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # 1. PS vs Outcome (heterogeneity check)
        ax = axes[0, 0]
        ax.scatter(ps[T == 1], Y[T == 1], alpha=0.3, color='coral', s=20, label='Treated')
        ax.scatter(ps[T == 0], Y[T == 0], alpha=0.3, color='steelblue', s=20, label='Control')
        ax.set_xlabel('Propensity Score')
        ax.set_ylabel('Outcome')
        ax.set_title('PS vs Outcome')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Common support region
        ax = axes[0, 1]
        ps_min_t = np.min(ps[T == 1])
        ps_max_t = np.max(ps[T == 1])
        ps_min_c = np.min(ps[T == 0])
        ps_max_c = np.max(ps[T == 0])
        
        common_min = max(ps_min_t, ps_min_c)
        common_max = min(ps_max_t, ps_max_c)
        
        ax.barh(['Control', 'Treated', 'Common'], 
               [ps_max_c - ps_min_c, ps_max_t - ps_min_t, common_max - common_min],
               left=[ps_min_c, ps_min_t, common_min],
               color=['steelblue', 'coral', 'purple'])
        ax.set_xlabel('Propensity Score Range')
        ax.set_title('Common Support Region')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # 3. Weight distribution
        ax = axes[0, 2]
        if weights is not None:
            ax.hist(weights[T == 1], bins=30, alpha=0.5, color='coral', label='Treated')
            ax.hist(weights[T == 0], bins=30, alpha=0.5, color='steelblue', label='Control')
            ax.set_xlabel('Weights')
            ax.set_ylabel('Frequency')
            ax.set_title('Weight Distribution')
            ax.legend()
            
            # 極端な重みの警告
            if np.max(weights) > 10:
                ax.text(0.5, 0.9, '⚠️ Extreme weights detected!',
                       transform=ax.transAxes, fontsize=12, color='red',
                       ha='center', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No weights provided', transform=ax.transAxes,
                   ha='center', va='center')
        ax.grid(True, alpha=0.3)
        
        # 4. Effective Sample Size by PS bins
        ax = axes[1, 0]
        ps_bins = np.linspace(0, 1, 11)
        bin_ess = []
        bin_centers = []
        
        for i in range(len(ps_bins) - 1):
            mask = (ps >= ps_bins[i]) & (ps < ps_bins[i + 1])
            if np.sum(mask) > 0 and weights is not None:
                w_bin = weights[mask]
                ess_bin = np.sum(w_bin)**2 / np.sum(w_bin**2)
                bin_ess.append(ess_bin / np.sum(mask))  # ESS ratio
                bin_centers.append((ps_bins[i] + ps_bins[i + 1]) / 2)
        
        if bin_ess:
            ax.bar(bin_centers, bin_ess, width=0.08, alpha=0.7, color='purple')
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Propensity Score Bin')
            ax.set_ylabel('ESS Ratio')
            ax.set_title('Effective Sample Size by PS Bin')
        ax.grid(True, alpha=0.3)
        
        # 5. Positivity violations
        ax = axes[1, 1]
        violations = []
        thresholds = [0.01, 0.05, 0.10, 0.15, 0.20]
        
        for thresh in thresholds:
            n_violate = np.sum((ps < thresh) | (ps > 1 - thresh))
            violations.append(100 * n_violate / len(ps))
        
        ax.plot(thresholds, violations, marker='o', linewidth=2, markersize=8)
        ax.fill_between(thresholds, 0, violations, alpha=0.3)
        ax.set_xlabel('Trimming Threshold')
        ax.set_ylabel('% Samples Excluded')
        ax.set_title('Positivity Violation Analysis')
        ax.grid(True, alpha=0.3)
        
        # 6. Summary statistics table
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')
        
        summary_data = [
            ['Metric', 'Value'],
            ['N Treated', f'{np.sum(T):.0f}'],
            ['N Control', f'{np.sum(1-T):.0f}'],
            ['Min PS', f'{np.min(ps):.4f}'],
            ['Max PS', f'{np.max(ps):.4f}'],
            ['PS < 0.01', f'{np.sum(ps < 0.01):.0f}'],
            ['PS > 0.99', f'{np.sum(ps > 0.99):.0f}'],
            ['Common Support', f'{common_min:.3f} - {common_max:.3f}']
        ]
        
        if weights is not None:
            ess_total = np.sum(weights)**2 / np.sum(weights**2)
            summary_data.append(['ESS', f'{ess_total:.0f} ({100*ess_total/len(Y):.1f}%)'])
            summary_data.append(['Max Weight', f'{np.max(weights):.2f}'])
        
        table = ax.table(cellText=summary_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.suptitle('Comprehensive Overlap Diagnostics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _calculate_smd(
        self,
        x: np.ndarray,
        t: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """標準化平均差を計算"""
        if weights is None:
            weights = np.ones_like(t)
        
        # 重み付き平均
        w1 = weights * t / np.sum(weights * t)
        w0 = weights * (1 - t) / np.sum(weights * (1 - t))
        
        mean1 = np.sum(x * w1)
        mean0 = np.sum(x * w0)
        
        # プールされた標準偏差
        var1 = np.sum(w1 * (x - mean1)**2)
        var0 = np.sum(w0 * (x - mean0)**2)
        pooled_std = np.sqrt((var1 + var0) / 2)
        
        if pooled_std == 0:
            return 0
        
        return (mean1 - mean0) / pooled_std
    
    def create_diagnostic_report(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        ps: np.ndarray,
        weights: np.ndarray,
        ate: float,
        ci: Tuple[float, float]
    ) -> None:
        """
        統合診断レポートの生成
        NASA JPL標準：意思決定の完全な証跡
        """
        print("="*80)
        print("CAUSAL INFERENCE DIAGNOSTIC REPORT")
        print("NASA JPL Standard Compliance Check")
        print("="*80)
        
        # 1. Sample Size Check
        n_total = len(Y)
        n_treated = np.sum(T)
        n_control = n_total - n_treated
        
        print(f"\n1. SAMPLE SIZE")
        print(f"   Total: {n_total}")
        print(f"   Treated: {n_treated} ({100*n_treated/n_total:.1f}%)")
        print(f"   Control: {n_control} ({100*n_control/n_total:.1f}%)")
        
        if min(n_treated, n_control) < 30:
            print("   ⚠️  WARNING: Small sample size detected")
        else:
            print("   ✓  Sample size adequate")
        
        # 2. Overlap Check
        print(f"\n2. OVERLAP ASSESSMENT")
        n_extreme = np.sum((ps < 0.01) | (ps > 0.99))
        print(f"   Extreme PS (<0.01 or >0.99): {n_extreme} ({100*n_extreme/n_total:.1f}%)")
        
        if n_extreme / n_total > 0.1:
            print("   ⚠️  WARNING: Significant positivity violations")
        else:
            print("   ✓  Positivity assumption reasonable")
        
        # 3. Balance Check
        print(f"\n3. COVARIATE BALANCE")
        max_smd = 0
        for i in range(X.shape[1]):
            smd = abs(self._calculate_smd(X[:, i], T, weights))
            max_smd = max(max_smd, smd)
        
        print(f"   Max SMD: {max_smd:.3f}")
        if max_smd > 0.1:
            print("   ⚠️  WARNING: Imbalance detected")
        else:
            print("   ✓  Covariates well balanced")
        
        # 4. ESS Check
        print(f"\n4. EFFECTIVE SAMPLE SIZE")
        ess = np.sum(weights)**2 / np.sum(weights**2)
        ess_ratio = ess / n_total
        
        print(f"   ESS: {ess:.0f} ({100*ess_ratio:.1f}% of original)")
        if ess_ratio < 0.2:
            print("   ⚠️  WARNING: Low ESS - consider alternative methods")
        else:
            print("   ✓  ESS adequate")
        
        # 5. Results
        print(f"\n5. CAUSAL ESTIMATE")
        print(f"   ATE: {ate:.4f}")
        print(f"   95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        
        if ci[0] > 0:
            print("   ✓  Significant positive effect")
        elif ci[1] < 0:
            print("   ✓  Significant negative effect")
        else:
            print("   ○  No significant effect detected")
        
        print("\n" + "="*80)
        print("END OF DIAGNOSTIC REPORT")
        print("="*80)


# デモンストレーション関数
def demonstrate_diagnostics():
    """診断ツールの実演"""
    print("Causal Diagnostics Demonstration")
    print("="*60)
    
    np.random.seed(42)
    
    # データ生成
    n = 2000
    p = 5
    X = np.random.randn(n, p)
    
    # PSを意図的に偏らせる
    logit_ps = 2 * X[:, 0] + X[:, 1] + 0.5 * X[:, 2]
    ps = 1 / (1 + np.exp(-logit_ps))
    T = np.random.binomial(1, ps)
    
    # アウトカム生成
    Y = X[:, 0] + 0.5 * T + 0.3 * X[:, 1] + np.random.randn(n)
    
    # IPW weights計算
    from sklearn.linear_model import LogisticRegression
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, T)
    ps_est = ps_model.predict_proba(X)[:, 1]
    ps_trimmed = np.clip(ps_est, 0.01, 0.99)
    
    # Stabilized weights
    p_treat = np.mean(T)
    weights = np.where(
        T == 1,
        p_treat / ps_trimmed,
        (1 - p_treat) / (1 - ps_trimmed)
    )
    
    # 診断ツールの実行
    diag = CausalDiagnostics()
    
    print("\n1. Creating Love Plot...")
    balance_stats = diag.create_love_plot(X, T, weights)
    
    print("\n2. PS Distribution Analysis...")
    ps_stats = diag.plot_propensity_score_distribution(ps_est, T, trim_threshold=0.01)
    
    print("\n3. Comprehensive Overlap Diagnostics...")
    diag.overlap_diagnostic_plot(ps_est, T, Y, weights)
    
    # ATE計算（簡易版）
    w1 = weights * T / np.sum(weights * T)
    w0 = weights * (1 - T) / np.sum(weights * (1 - T))
    ate = np.sum(Y * w1) - np.sum(Y * w0)
    
    # 簡易的な信頼区間
    se = np.sqrt(np.var(Y * w1 - Y * w0) / n)
    ci = (ate - 1.96 * se, ate + 1.96 * se)
    
    print("\n4. Generating Diagnostic Report...")
    diag.create_diagnostic_report(X, T, Y, ps_est, weights, ate, ci)


if __name__ == "__main__":
    demonstrate_diagnostics()
