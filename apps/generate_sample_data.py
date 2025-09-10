# apps/generate_sample_data.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_sample_push_data(n_samples=10000, seed=42):
    """
    Push配信効果分析用のサンプルデータを生成
    
    GoogleのA/Bテストシミュレーターを参考に実装
    現実的な効果サイズとノイズを含む
    """
    np.random.seed(seed)
    
    print("="*60)
    print("📊 サンプルデータ生成")
    print("="*60)
    
    # 1. 顧客特徴量の生成
    print("\n1️⃣ 顧客特徴量を生成中...")
    
    # 年齢（正規分布、20-70歳）
    age = np.clip(np.random.normal(40, 15, n_samples), 20, 70).astype(int)
    
    # 性別（0: 女性, 1: 男性）
    gender_male = np.random.binomial(1, 0.45, n_samples)
    
    # RFM特徴量
    # Recency: 最終購買からの日数（指数分布）
    recency_days = np.clip(np.random.exponential(10, n_samples), 1, 365).astype(int)
    
    # Frequency: 30日間の購買回数（ポアソン分布）
    frequency_30d = np.random.poisson(5, n_samples)
    
    # Monetary: 30日間の購買金額（対数正規分布）
    monetary_30d = np.exp(np.random.normal(8, 1.5, n_samples))
    
    # 過去の購買履歴（CUPED用）
    prev_week_sales = np.exp(np.random.normal(7, 1.5, n_samples))
    prev_month_sales = np.exp(np.random.normal(8.5, 1.5, n_samples))
    
    # 2. 傾向スコアの真値を計算（観測されない）
    print("2️⃣ 処置割り当てを生成中...")
    
    # ロジスティック回帰モデルで傾向スコアを決定
    # 若い人、頻繁に買う人ほどPush配信される確率が高い
    logit_ps = (
        -2.0 +  # 切片
        -0.02 * (age - 40) +  # 年齢効果
        0.1 * frequency_30d +  # 頻度効果
        0.3 * (monetary_30d > np.median(monetary_30d)) +  # 高額購入者
        0.2 * gender_male  # 性別効果
    )
    
    # 真の傾向スコア
    true_propensity = 1 / (1 + np.exp(-logit_ps))
    
    # 処置割り当て（Push配信有無）
    treated = np.random.binomial(1, true_propensity)
    
    print(f"   処置群: {treated.sum():,} ({treated.mean()*100:.1f}%)")
    print(f"   統制群: {(1-treated).sum():,} ({(1-treated).mean()*100:.1f}%)")
    
    # 3. 潜在結果の生成
    print("3️⃣ 購買結果を生成中...")
    
    # ベースラインの購買額（処置なしの場合）
    baseline_sales = (
        1000 +  # 基本購買額
        50 * frequency_30d +  # 頻度による増加
        0.1 * monetary_30d +  # 過去の購買額の影響
        -10 * (age - 40) +  # 年齢効果
        500 * gender_male +  # 性別効果
        0.3 * prev_week_sales +  # 自己相関
        np.random.normal(0, 500, n_samples)  # ノイズ
    )
    
    # 処置効果（異質効果を含む）
    # 若い人、頻繁に買う人ほど効果が大きい
    individual_treatment_effect = (
        500 +  # 平均処置効果
        -5 * (age - 40) +  # 年齢による効果の違い
        20 * frequency_30d +  # 頻度による効果の違い
        np.random.normal(0, 200, n_samples)  # 個人差
    )
    
    # 観測される購買額
    outcome_sales = baseline_sales + treated * individual_treatment_effect
    outcome_sales = np.maximum(0, outcome_sales)  # 負の値を0に
    
    # 購買個数（購買額と相関）
    outcome_quantity = np.maximum(
        0,
        np.round(outcome_sales / 1000 + np.random.normal(0, 1, n_samples))
    ).astype(int)
    
    # 4. データフレームの作成
    print("4️⃣ データフレームを構築中...")
    
    df = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'treated': treated,
        'outcome_sales': outcome_sales,
        'outcome_quantity': outcome_quantity,
        'age': age,
        'gender_male': gender_male,
        'recency_days': recency_days,
        'frequency_30d': frequency_30d,
        'monetary_30d': monetary_30d,
        'prev_week_sales': prev_week_sales,
        'prev_month_sales': prev_month_sales,
        # 診断用（実際の分析では使わない）
        '_true_propensity': true_propensity,
        '_true_effect': individual_treatment_effect
    })
    
    # 5. データの要約統計
    print("\n📈 データ要約統計:")
    print("-"*40)
    
    # 処置群と統制群の平均を比較
    treated_mean = df[df['treated']==1]['outcome_sales'].mean()
    control_mean = df[df['treated']==0]['outcome_sales'].mean()
    naive_ate = treated_mean - control_mean
    
    print(f"処置群の平均購買額: {treated_mean:,.2f} 円")
    print(f"統制群の平均購買額: {control_mean:,.2f} 円")
    print(f"単純な差（バイアスあり）: {naive_ate:,.2f} 円")
    
    # 真の平均処置効果（シミュレーションなので分かる）
    true_ate = df['_true_effect'].mean()
    print(f"真の平均処置効果: {true_ate:,.2f} 円")
    print(f"選択バイアス: {naive_ate - true_ate:,.2f} 円")
    
    return df

def save_sample_data(df, output_dir='data'):
    """
    サンプルデータを保存
    """
    # ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 分析用データ（真値を除外）
    analysis_cols = [col for col in df.columns if not col.startswith('_')]
    df_analysis = df[analysis_cols]
    
    # CSV形式で保存（確認用）
    csv_path = os.path.join(output_dir, 'sample_data.csv')
    df_analysis.to_csv(csv_path, index=False)
    print(f"\n💾 CSVファイル保存: {csv_path}")
    
    # Parquet形式で保存（分析用）
    parquet_path = os.path.join(output_dir, 'sample_data.parquet')
    df_analysis.to_parquet(parquet_path, index=False)
    print(f"💾 Parquetファイル保存: {parquet_path}")
    
    # 真値を含む完全データ（検証用）
    full_parquet_path = os.path.join(output_dir, 'sample_data_with_truth.parquet')
    df.to_parquet(full_parquet_path, index=False)
    print(f"💾 検証用データ保存: {full_parquet_path}")
    
    return csv_path, parquet_path

if __name__ == "__main__":
    # メイン実行
    print("\n" + "="*60)
    print("🚀 Push配信効果分析用サンプルデータ生成")
    print("   Google/Meta水準のシミュレーション")
    print("="*60)
    
    # データ生成
    df = generate_sample_push_data(n_samples=10000)
    
    # データ保存
    csv_path, parquet_path = save_sample_data(df)
    
    print("\n✅ サンプルデータ生成完了！")
    print("\n次のステップ:")
    print("1. データ確認: head data/sample_data.csv")
    print("2. 分析実行: python apps/run_ate_analysis.py")
