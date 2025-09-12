import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 自作ライブラリ
from libs.causal.dr_ate import DRATEEstimator, test_dr_ate_estimator
from data.synthetic.generate_sample_data import KARTEDataSimulator

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class PushATEAnalyzer:
    """Push配信効果の包括的ATE分析器"""
    
    def __init__(self, data_path: str = "data/synthetic/tdf_v1.parquet"):
        """
        Parameters:
        -----------
        data_path : str
            分析対象データのパス
        """
        self.data_path = data_path
        self.results = {}
        self.diagnostics = {}
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """データ読み込みと前処理"""
        
        if not Path(self.data_path).exists():
            print("サンプルデータが見つかりません。生成中...")
            simulator = KARTEDataSimulator(n_customers=10000)
            data = simulator.generate_all_data()
            # データ保存は既にシミュレータ内で実行済み
            print("サンプルデータ生成完了")
            
        # データ読み込み
        tdf = pd.read_parquet(self.data_path)
        print(f"データ読み込み完了: {tdf.shape}")
        
        # 基本統計
        print("\n=== データサマリー ===")
        print(f"サンプル数: {len(tdf):,}")
        print(f"処置率（Push配信率）: {tdf['treatment'].mean():.3f}")
        print(f"平均購買額（全体）: {tdf['total_amount'].mean():.0f}円")
        print(f"平均購買額（処置群）: {tdf[tdf['treatment']==1]['total_amount'].mean():.0f}円")
        print(f"平均購買額（対照群）: {tdf[tdf['treatment']==0]['total_amount'].mean():.0f}円")
        
        # 基本的な差分（単純比較、因果効果ではない）
        simple_diff = (tdf[tdf['treatment']==1]['total_amount'].mean() - 
                      tdf[tdf['treatment']==0]['total_amount'].mean())
        print(f"単純平均差分: {simple_diff:.0f}円")
        print("※この値は選択バイアスを含むため、真の因果効果ではない")
        
        return tdf
    
    def prepare_features(self, tdf: pd.DataFrame) -> tuple:
        """特徴量の準備"""
        
        # 特徴量選択（顧客属性）
        feature_cols = [
            'age',                    # 年齢
            'gender_M',               # 性別（男性=1）
            'income_high',            # 高収入フラグ
            'income_very_high',       # 超高収入フラグ  
            'app_usage_days',         # アプリ使用日数
            'segment_premium',        # プレミアム顧客フラグ
            'segment_regular',        # 定期購入顧客フラグ
            'segment_occasional'      # 時々購入顧客フラグ
        ]
        
        X = tdf[feature_cols].values
        T = tdf['treatment'].values  # Push配信フラグ
        Y = tdf['total_amount'].values  # 購買金額
        
        print(f"\n特徴量数: {X.shape[1]}")
        print("特徴量リスト:", feature_cols)
        
        # 特徴量の基本統計
        feature_stats = tdf[feature_cols].describe()
        print("\n=== 特徴量統計 ===")
        print(feature_stats.round(3))
        
        return X, T, Y, feature_cols
    
    def run_dr_ate_analysis(self, X, T, Y) -> dict:
        """DR-ATE分析の実行"""
        
        print("\n=== DR-ATE推定実行中 ===")
        
        # クロスフィット版
        print("1. クロスフィット版DR推定...")
        estimator_cf = DRATEEstimator(
            outcome_model='random_forest',
            propensity_model='logistic',
            cross_fit=True,
            n_folds=5,
            random_state=42
        )
        estimator_cf.fit(X, T, Y)
        results_cf = estimator_cf.summary()
        
        # 非クロスフィット版（比較用）
        print("2. 標準版DR推定...")
        estimator_std = DRATEEstimator(
            outcome_model='random_forest',
            propensity_model='logistic', 
            cross_fit=False,
            random_state=42
        )
        estimator_std.fit(X, T, Y)
        results_std = estimator_std.summary()
        
        # 結果比較
        print("\n=== ATE推定結果比較 ===")
        comparison = pd.DataFrame({
            'method': ['Cross-fit DR', 'Standard DR'],
            'ate': [results_cf['ate'], results_std['ate']],
            'se': [results_cf['se'], results_std['se']],
            'ci_lower': [results_cf['ci_lower'], results_std['ci_lower']],
            'ci_upper': [results_cf['ci_upper'], results_std['ci_upper']],
            'p_value': [results_cf['p_value'], results_std['p_value']]
        })
        
        print(comparison.round(3))
        
        return {
            'cross_fit': results_cf,
            'standard': results_std,
            'comparison': comparison,
            'estimator_cf': estimator_cf,
            'estimator_std': estimator_std
        }
    
    def generate_diagnostics(self, results: dict, X, T, Y) -> dict:
        """診断統計の生成"""
        
        print("\n=== 診断統計 ===")
        
        estimator = results['estimator_cf']  # クロスフィット版を使用
        diagnostics = estimator.diagnostics_
        
        # 基本診断
        print(f"サンプル数: {diagnostics['n_samples']:,}")
        print(f"特徴量数: {diagnostics['n_features']}")
        print(f"処置率: {diagnostics['treatment_rate']:.3f}")
        print(f"傾向スコア平均: {diagnostics['ps_mean']:.3f}")
        print(f"傾向スコア標準偏差: {diagnostics['ps_std']:.3f}")
        print(f"裾切り率: {diagnostics['trim_rate']:.3f}")
        
        # バランス診断（処置群・対照群の共変量分布比較）
        balance_stats = self._calculate_balance_stats(X, T)
        
        # Overlap診断（共通サポートの確認）
        overlap_stats = self._calculate_overlap_stats(X, T)
        
        return {
            'basic': diagnostics,
            'balance': balance_stats,
            'overlap': overlap_stats
        }
    
    def _calculate_balance_stats(self, X, T):
        """バランス統計の計算"""
        
        # 標準化平均差分（Standardized Mean Difference）
        X_treated = X[T == 1]
        X_control = X[T == 0]
        
        mean_treated = np.mean(X_treated, axis=0)
        mean_control = np.mean(X_control, axis=0)
        std_pooled = np.sqrt((np.var(X_treated, axis=0) + np.var(X_control, axis=0)) / 2)
        
        smd = (mean_treated - mean_control) / std_pooled
        
        balance_stats = {
            'smd': smd,
            'smd_max': np.max(np.abs(smd)),
            'smd_mean': np.mean(np.abs(smd)),
            'imbalance_features': np.sum(np.abs(smd) > 0.1)  # |SMD| > 0.1は不均衡
        }
        
        print(f"最大SMD: {balance_stats['smd_max']:.3f}")
        print(f"平均SMD: {balance_stats['smd_mean']:.3f}")
        print(f"不均衡特徴量数: {balance_stats['imbalance_features']}")
        
        return balance_stats
    
    def _calculate_overlap_stats(self, X, T):
        """オーバーラップ統計の計算"""
        
        from sklearn.linear_model import LogisticRegression
        
        # 傾向スコアを再計算（診断用）
        ps_model = LogisticRegression(random_state=42)
        ps_model.fit(X, T)
        ps = ps_model.predict_proba(X)[:, 1]
        
        overlap_stats = {
            'ps_min': np.min(ps),
            'ps_max': np.max(ps),
            'ps_extreme_low': np.mean(ps < 0.05),  # 極端に低い傾向スコア
            'ps_extreme_high': np.mean(ps > 0.95),  # 極端に高い傾向スコア
            'common_support_rate': np.mean((ps >= 0.05) & (ps <= 0.95))
        }
        
        print(f"傾向スコア範囲: [{overlap_stats['ps_min']:.3f}, {overlap_stats['ps_max']:.3f}]")
        print(f"極端傾向スコア率: {overlap_stats['ps_extreme_low'] + overlap_stats['ps_extreme_high']:.3f}")
        print(f"共通サポート率: {overlap_stats['common_support_rate']:.3f}")
        
        return overlap_stats
    
    def create_visualizations(self, tdf, results, feature_cols):
        """可視化の生成"""
        
        print("\n=== 可視化生成中 ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ATE推定値比較
        comparison = results['comparison']
        ax1 = axes[0, 0]
        methods = comparison['method']
        ates = comparison['ate']
        errors = comparison['se'] * 1.96  # 95%CI
        
        ax1.errorbar(methods, ates, yerr=errors, fmt='o', capsize=5, capthick=2)
        ax1.set_title('ATE推定値比較（95%信頼区間）', fontsize=14)
        ax1.set_ylabel('ATE推定値（円）', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. 処置群・対照群の分布比較
        ax2 = axes[0, 1]
        tdf[tdf['treatment']==1]['total_amount'].hist(alpha=0.7, bins=50, label='処置群（Push配信）', ax=ax2)
        tdf[tdf['treatment']==0]['total_amount'].hist(alpha=0.7, bins=50, label='対照群（非配信）', ax=ax2)
        ax2.set_title('購買額分布比較', fontsize=14)
        ax2.set_xlabel('購買額（円）', fontsize=12)
        ax2.set_ylabel('頻度', fontsize=12)
        ax2.legend()
        
        # 3. セグメント別ATE（簡易版）
        ax3 = axes[1, 0]
        segment_ate = []
        segments = ['premium', 'regular', 'occasional', 'inactive']
        
        for segment in segments:
            seg_data = tdf[tdf['segment'] == segment]
            if len(seg_data) > 100:  # 十分なサンプルサイズ
                treated_mean = seg_data[seg_data['treatment']==1]['total_amount'].mean()
                control_mean = seg_data[seg_data['treatment']==0]['total_amount'].mean()
                segment_ate.append(treated_mean - control_mean)
            else:
                segment_ate.append(np.nan)
        
        ax3.bar(segments, segment_ate, alpha=0.7)
        ax3.set_title('セグメント別単純平均差分', fontsize=14)
        ax3.set_ylabel('平均差分（円）', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 影響関数分布
        ax4 = axes[1, 1]
        if_values = results['estimator_cf'].influence_function_
        ax4.hist(if_values, bins=50, alpha=0.7, edgecolor='black')
        ax4.set_title('影響関数分布', fontsize=14)
        ax4.set_xlabel('影響関数値', fontsize=12)
        ax4.set_ylabel('頻度', fontsize=12)
        ax4.axvline(0, color='red', linestyle='--', alpha=0.7, label='平均=0')
        ax4.legend()
        
        plt.tight_layout()
        
        # 保存
        os.makedirs('docs/figures', exist_ok=True)
        plt.savefig('docs/figures/push_ate_analysis.png', dpi=300, bbox_inches='tight')
        print("可視化を docs/figures/push_ate_analysis.png に保存しました")
        
        plt.show()
    
    def generate_report(self, results, diagnostics):
        """分析レポートの生成"""
        
        print("\n" + "="*60)
        print("Push配信効果ATE分析レポート")
        print("="*60)
        
        cf_results = results['cross_fit']
        
        print(f"\n【主要結果】")
        print(f"・ATE推定値: {cf_results['ate']:.2f}円")
        print(f"・標準誤差: {cf_results['se']:.2f}円") 
        print(f"・95%信頼区間: [{cf_results['ci_lower']:.2f}, {cf_results['ci_upper']:.2f}]円")
        print(f"・p値: {cf_results['p_value']:.6f}")
        print(f"・統計的有意性: {'有意' if cf_results['significant'] else '非有意'}（α=0.05）")
        
        print(f"\n【ビジネス解釈】")
        if cf_results['significant']:
            roi_estimate = cf_results['ate'] / 100  # 仮にPush配信コスト100円として
            print(f"・Push配信により1人当たり平均{cf_results['ate']:.0f}円の購買増加")
            print(f"・配信コスト100円と仮定した場合のROI: {roi_estimate:.2f}")
            print(f"・95%の確率で真の効果は{cf_results['ci_lower']:.0f}円～{cf_results['ci_upper']:.0f}円の範囲")
        else:
            print(f"・統計的に有意な効果は検出されませんでした")
            print(f"・ただし、信頼区間内には実用的な効果サイズも含まれる可能性があります")
        
        print(f"\n【診断結果】")
        basic = diagnostics['basic']
        balance = diagnostics['balance']
        overlap = diagnostics['overlap']
        
        print(f"・サンプル数: {basic['n_samples']:,}人")
        print(f"・処置率: {basic['treatment_rate']:.1%}")
        print(f"・最大共変量不均衡: {balance['smd_max']:.3f}")
        print(f"・共通サポート率: {overlap['common_support_rate']:.1%}")
        
        # 品質評価
        quality_flags = []
        if basic['n_samples'] >= 1000:
            quality_flags.append("✓ 十分なサンプルサイズ")
        if 0.1 <= basic['treatment_rate'] <= 0.9:
            quality_flags.append("✓ 適切な処置率")
        if balance['smd_max'] <= 0.25:
            quality_flags.append("✓ 良好な共変量バランス")
        if overlap['common_support_rate'] >= 0.9:
            quality_flags.append("✓ 良好な共通サポート")
            
        print(f"\n【品質チェック】")
        for flag in quality_flags:
            print(f"{flag}")
            
        print(f"\n【推奨事項】")
        if cf_results['significant']:
            print("・Push配信の継続実施を推奨")
            print("・セグメント別の詳細分析で効果の異質性を確認")
            print("・A/Bテストによる確認実験の実施を検討")
        else:
            print("・追加データによる分析の継続")
            print("・配信戦略（タイミング、内容）の見直し")
            print("・セグメント絞り込みによる効果向上の検討")
            
        return cf_results

def main():
    """メイン分析実行"""
    
    print("Push配信効果ATE分析を開始します...\n")
    
    # 分析器初期化
    analyzer = PushATEAnalyzer()
    
    # データ読み込み・準備
    tdf = analyzer.load_and_prepare_data()
    X, T, Y, feature_cols = analyzer.prepare_features(tdf)
    
    # DR-ATE分析実行
    results = analyzer.run_dr_ate_analysis(X, T, Y)
    
    # 診断統計
    diagnostics = analyzer.generate_diagnostics(results, X, T, Y)
    
    # 可視化
    analyzer.create_visualizations(tdf, results, feature_cols)
    
    # レポート生成
    final_results = analyzer.generate_report(results, diagnostics)
    
    print(f"\n分析完了！結果は docs/figures/ に保存されました。")
    
    return analyzer, results, diagnostics

if __name__ == "__main__":
    analyzer, results, diagnostics = main()
