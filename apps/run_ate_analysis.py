# apps/run_ate_analysis.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# scikit-learnのインポート（明示的にフルネーム使用）
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 自作モジュール
from libs.causal.dr_ate import DoubleRobustATE

def load_data(data_path: str = 'data/sample_data.parquet') -> pd.DataFrame:
    """
    データの読み込み
    
    Args:
        data_path: データファイルのパス
    
    Returns:
        DataFrame
    """
    if not os.path.exists(data_path):
        print(f"⚠️ データファイルが見つかりません: {data_path}")
        print("まず以下を実行してください:")
        print("  python apps/generate_sample_data.py")
        sys.exit(1)
    
    print(f"📂 データ読み込み: {data_path}")
    
    # 拡張子によって読み込み方法を変更
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    print(f"✅ {len(df):,} 件のデータを読み込みました")
    
    return df

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    特徴量の準備
    
    Args:
        df: データフレーム
    
    Returns:
        X, T, Y の tuple
    """
    # 特徴量カラム
    feature_cols = [
        'age', 'gender_male', 'recency_days', 
        'frequency_30d', 'monetary_30d', 
        'prev_week_sales', 'prev_month_sales'
    ]
    
    # 欠損値の確認
    missing = df[feature_cols].isnull().sum()
    if missing.sum() > 0:
        print("\n⚠️ 欠損値が見つかりました:")
        print(missing[missing > 0])
        print("欠損値を中央値で補完します...")
        
        for col in feature_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
    
    X = df[feature_cols].values
    T = df['treated'].values
    Y = df['outcome_sales'].values
    
    print(f"\n📊 データ形状:")
    print(f"   X (特徴量): {X.shape}")
    print(f"   T (処置): {T.shape}")
    print(f"   Y (結果): {Y.shape}")
    
    return X, T, Y

def create_diagnostic_plots(dr_estimator, save_dir='docs'):
    """
    診断プロットの作成
    
    Args:
        dr_estimator: 学習済みのDR推定器
        save_dir: 保存先ディレクトリ
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # スタイル設定
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 傾向スコアの分布
    ax = axes[0, 0]
    ps = dr_estimator.e_hat
    treated_mask = dr_estimator.ate_result.n_treated
    
    ax.hist(ps[dr_estimator.ate_result.n_treated], 
            alpha=0.6, label='Treated', bins=30, color='orange')
    ax.hist(ps[~dr_estimator.ate_result.n_treated], 
            alpha=0.6, label='Control', bins=30, color='blue')
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Propensity Score Distribution')
    ax.legend()
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5)
    
    # 2. 影響関数の分布
    ax = axes[0, 1]
    psi = dr_estimator.ate_result.influence_function
    ate = dr_estimator.ate_result.ate
    
    ax.hist(psi, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(ate, color='red', linestyle='--', 
               label=f'ATE={ate:.2f}', linewidth=2)
    ax.set_xlabel('Influence Function Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Influence Function Distribution')
    ax.legend()
    
    # 3. 予測値 vs 実測値（処置群）
    ax = axes[0, 2]
    mu_1 = dr_estimator.mu_1_hat
    ax.scatter(mu_1[:100], psi[:100], alpha=0.5, s=20)
    ax.set_xlabel('Predicted Outcome (Treated)')
    ax.set_ylabel('Influence Function')
    ax.set_title('Prediction vs Influence (Sample)')
    
    # 4. 傾向スコア vs 影響関数
    ax = axes[1, 0]
    ax.scatter(ps, psi, alpha=0.3, s=10)
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Influence Function')
    ax.set_title('PS vs Influence Function')
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.3)
    ax.axhline(ate, color='red', linestyle='--', alpha=0.3)
    
    # 5. 重みの分布
    ax = axes[1, 1]
    weights = np.where(dr_estimator.ate_result.n_treated, 
                      1/ps, 1/(1-ps))
    ax.hist(np.clip(weights, 0, 20), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('IPW Weights (clipped at 20)')
    ax.set_ylabel('Frequency')
    ax.set_title('Weight Distribution')
    
    # 6. 診断統計量のテキスト
    ax = axes[1, 2]
    ax.axis('off')
    
    diagnostics = dr_estimator.get_diagnostics()
    
    stats_text = f"""
    Diagnostic Statistics
    ─────────────────────────
    
    Propensity Score:
    • Range: [{diagnostics['propensity_score']['min']:.3f}, 
             {diagnostics['propensity_score']['max']:.3f}]
    • Mean: {diagnostics['propensity_score']['mean']:.3f}
    • Extreme (<0.1 or >0.9): 
      {diagnostics['propensity_score']['extreme_low']*100:.1f}%
    
    Effective Sample Size:
    • ESS Ratio: {diagnostics['ess_ratio']:.3f}
    • Max Weight: {diagnostics['max_weight']:.1f}
    
    Quality Check:
    {'✅ Good balance' if diagnostics['ess_ratio'] > 0.2 else '⚠️ Poor balance'}
    {'✅ Stable weights' if diagnostics['max_weight'] < 50 else '⚠️ Extreme weights'}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, 
            verticalalignment='center', fontfamily='monospace')
    
    plt.suptitle('Double Robust ATE - Diagnostic Report', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存
    plot_path = os.path.join(save_dir, 'dr_diagnostics.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 診断プロット保存: {plot_path}")
    
    plt.show()

def main():
    """
    メイン実行関数
    """
    print("\n" + "="*80)
    print("🚀 Push配信効果のダブルロバスト推定")
    print("   サンプルデータを使用した分析")
    print("="*80)
    
    # 1. データ読み込み
    print("\n" + "─"*60)
    print("📥 Phase 1: データ読み込み")
    print("─"*60)
    
    df = load_data('data/sample_data.parquet')
    
    # データの概要表示
    print("\nデータ概要:")
    print(df.info())
    
    print("\n基本統計:")
    print(df[['treated', 'outcome_sales', 'age', 'frequency_30d']].describe())
    
    # 2. 特徴量準備
    print("\n" + "─"*60)
    print("🔧 Phase 2: 特徴量準備")
    print("─"*60)
    
    X, T, Y = prepare_features(df)
    
    # 標準化
    print("\n📏 特徴量を標準化中...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. モデル設定
    print("\n" + "─"*60)
    print("⚙️ Phase 3: モデル設定")
    print("─"*60)
    
    # 結果予測モデル（Random Forest）
    outcome_model = RandomForestRegressor(
        n_estimators=100,      # 木の数
        max_depth=10,          # 木の深さ
        min_samples_leaf=50,   # 葉の最小サンプル数
        random_state=42,       # 乱数シード
        n_jobs=-1              # 並列処理
    )
    
    # 傾向スコアモデル（Random Forest分類器）
    propensity_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=100,
        random_state=42,
        n_jobs=-1
    )
    
    print("✅ モデル設定完了:")
    print(f"   結果モデル: {outcome_model.__class__.__name__}")
    print(f"   傾向スコアモデル: {propensity_model.__class__.__name__}")
    
    # 4. DR推定
    print("\n" + "─"*60)
    print("🧮 Phase 4: ダブルロバスト推定")
    print("─"*60)
    
    dr_estimator = DoubleRobustATE(
        outcome_model=outcome_model,
        propensity_model=propensity_model,
        trim_threshold=0.02,  # 2%でトリミング
        verbose=True
    )
    
    # 学習と推定
    dr_estimator.fit(X_scaled, T, Y)
    
    # 5. 結果の解釈
    print("\n" + "─"*60)
    print("📈 Phase 5: 結果の解釈")
    print("─"*60)
    
    result = dr_estimator.ate_result
    
    # 効果サイズの計算
    control_mean = Y[T == 0].mean()
    relative_effect = (result.ate / control_mean) * 100
    
    print(f"\n💡 ビジネス的解釈:")
    print(f"   統制群の平均購買額: {control_mean:,.2f} 円")
    print(f"   処置効果（絶対値）: {result.ate:,.2f} 円")
    print(f"   処置効果（相対値）: {relative_effect:.1f}%")
    
    # ROI計算（仮の配信コスト）
    cost_per_push = 10  # Push配信1件あたりのコスト
    roi = (result.ate - cost_per_push) / cost_per_push * 100
    
    print(f"\n💰 ROI分析:")
    print(f"   配信コスト: {cost_per_push} 円/件")
    print(f"   純利益: {result.ate - cost_per_push:,.2f} 円/件")
    print(f"   ROI: {roi:.1f}%")
    
    if roi > 0:
        print(f"   → ✅ Push配信は費用対効果が高い")
    else:
        print(f"   → ❌ Push配信は費用対効果が低い")
    
    # 6. 診断
    print("\n" + "─"*60)
    print("🔍 Phase 6: 診断と可視化")
    print("─"*60)
    
    diagnostics = dr_estimator.get_diagnostics()
    
    # 品質チェック
    quality_checks = {
        'ESS比 > 0.2': diagnostics['ess_ratio'] > 0.2,
        '最大重み < 50': diagnostics['max_weight'] < 50,
        '極端なPS < 10%': (diagnostics['propensity_score']['extreme_low'] + 
                          diagnostics['propensity_score']['extreme_high']) < 0.1
    }
    
    print("\n品質チェック:")
    for check, passed in quality_checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")
    
    # 診断プロット
    create_diagnostic_plots(dr_estimator)
    
    # 7. 結果の保存
    print("\n" + "─"*60)
    print("💾 Phase 7: 結果の保存")
    print("─"*60)
    
    # 結果をDataFrameに整理
    results_df = pd.DataFrame({
        'metric': ['ATE', 'SE', 'CI_lower', 'CI_upper', 
                  'Relative_effect_%', 'ROI_%', 'N_treated', 'N_control'],
        'value': [result.ate, result.se, result.ci_lower, result.ci_upper,
                 relative_effect, roi, result.n_treated, result.n_control]
    })
    
    # 保存
    os.makedirs('results', exist_ok=True)
    results_path = 'results/dr_ate_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"✅ 結果を保存: {results_path}")
    
    # 真値との比較（サンプルデータの場合のみ）
    if os.path.exists('data/sample_data_with_truth.parquet'):
        print("\n" + "─"*60)
        print("🎯 真値との比較（検証用）")
        print("─"*60)
        
        df_truth = pd.read_parquet('data/sample_data_with_truth.parquet')
        true_ate = df_truth['_true_effect'].mean()
        
        print(f"   真のATE: {true_ate:,.2f} 円")
        print(f"   推定ATE: {result.ate:,.2f} 円")
        print(f"   推定誤差: {abs(result.ate - true_ate):,.2f} 円")
        print(f"   相対誤差: {abs(result.ate - true_ate) / true_ate * 100:.1f}%")
        
        # 真値が信頼区間に含まれるか
        if result.ci_lower <= true_ate <= result.ci_upper:
            print(f"   ✅ 真値は95%信頼区間に含まれる")
        else:
            print(f"   ❌ 真値は95%信頼区間に含まれない")
    
    print("\n" + "="*80)
    print("✨ 分析完了！")
    print("="*80)

if __name__ == "__main__":
    main()
