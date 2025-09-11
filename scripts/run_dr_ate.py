from pathlib import Path
import sys
import pandas as pd
import numpy as np

# ==== ルート固定（このファイルの2階層上をプロジェクトルートとみなす）====
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))              # libs/ が import 可能に
DATA = ROOT / "data" / "synthetic"

# ==== ライブラリ ====
from libs.causal.dr_ate import DRATEEstimator  # ← 既存のファイル構成を想定

# ==== データ読み込み ====
tdf = pd.read_parquet(DATA / "tdf_v1.parquet")
print("=== データ概要 ===")
print("TDF shape:", tdf.shape)
print("Treatment rate:", tdf["treatment"].mean())

# ==== 特徴量 ====
feature_cols = [
    'age','gender_M','income_high','income_very_high',
    'app_usage_days','segment_premium','segment_regular','segment_occasional'
]
X = tdf[feature_cols].values
T = tdf['treatment'].values
Y = tdf['total_amount'].values

# ==== 推定 ====
estimator = DRATEEstimator(cross_fit=True, random_state=42)
estimator.fit(X, T, Y)
results = estimator.summary()

# ==== 出力 ====
print("\n== Push配信のATE推定結果 ==")
print(f"ATE推定値: {results['ate']:.2f}円")
print(f"標準誤差: {results['se']:.2f}円")
print(f"95%CI: [{results['ci_lower']:.2f}, {results['ci_upper']:.2f}]")
print(f"p値: {results['p_value']:.6f}")
print("統計的有意性:", "有意" if results["significant"] else "非有意")
print(f"処置率: {results['diagnostics']['treatment_rate']:.3f}")
