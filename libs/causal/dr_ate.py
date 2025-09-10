# libs/causal/dr_ate.py

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from scikit-learn.base import clone  # scikit-learnを明示的に使用

@dataclass
class DREstimatorResult:
    """DR推定結果を格納するクラス"""
    ate: float
    se: float  
    ci_lower: float
    ci_upper: float
    influence_function: np.ndarray
    n_treated: int
    n_control: int
    propensity_scores: np.ndarray
    outcome_predictions_1: np.ndarray
    outcome_predictions_0: np.ndarray
    
class DoubleRobustATE:
    """
    ダブルロバスト平均処置効果推定器
    
    参考実装:
    - Google Ads の Causal Impact Framework
    - Meta の Adaptive Experimentation Platform
    - Microsoft の EconML ライブラリ
    """
    
    def __init__(self, 
                 outcome_model=None,
                 propensity_model=None,
                 trim_threshold: float = 0.01,
                 verbose: bool = True):
        """
        Args:
            outcome_model: 結果予測モデル（scikit-learn互換）
            propensity_model: 傾向スコアモデル（scikit-learn互換）
            trim_threshold: 傾向スコアのトリミング閾値
            verbose: 詳細出力フラグ
        """
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.trim_threshold = trim_threshold
        self.verbose = verbose
        
        # 学習済みモデルを保存
        self.outcome_model_1 = None
        self.outcome_model_0 = None
        self.e_hat = None
        self.mu_1_hat = None
        self.mu_0_hat = None
        
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'DoubleRobustATE':
        """
        モデルの学習
        
        Args:
            X: 共変量行列 (n_samples, n_features)
            T: 処置インジケータ (n_samples,)
            Y: 結果変数 (n_samples,)
        """
        if self.verbose:
            print("\n" + "="*60)
            print("🔬 ダブルロバスト推定の実行")
            print("="*60)
        
        # 入力検証
        self._validate_inputs(X, T, Y)
        
        # 1. 傾向スコアの学習
        if self.verbose:
            print("\n📊 Step 1: 傾向スコアモデルの学習")
            print("-"*40)
        
        self.propensity_model.fit(X, T)
        
        # 傾向スコアの予測
        if hasattr(self.propensity_model, 'predict_proba'):
            self.e_hat = self.propensity_model.predict_proba(X)[:, 1]
        else:
            # predict_probaがない場合（例：線形回帰）
            self.e_hat = self.propensity_model.predict(X)
            self.e_hat = np.clip(self.e_hat, 0, 1)
        
        # トリミング（極端な傾向スコアを制限）
        original_e_hat = self.e_hat.copy()
        self.e_hat = np.clip(self.e_hat, self.trim_threshold, 1 - self.trim_threshold)
        
        if self.verbose:
            n_trimmed = np.sum((original_e_hat < self.trim_threshold) | 
                              (original_e_hat > 1 - self.trim_threshold))
            print(f"   傾向スコア範囲: [{self.e_hat.min():.3f}, {self.e_hat.max():.3f}]")
            print(f"   トリミングされた観測: {n_trimmed} ({n_trimmed/len(T)*100:.1f}%)")
        
        # 2. 結果モデルの学習（処置群と統制群で別々に）
        if self.verbose:
            print("\n📈 Step 2: 結果モデルの学習")
            print("-"*40)
        
        # 処置群モデル
        X_treated = X[T == 1]
        Y_treated = Y[T == 1]
        self.outcome_model_1 = clone(self.outcome_model)
        self.outcome_model_1.fit(X_treated, Y_treated)
        
        # 統制群モデル  
        X_control = X[T == 0]
        Y_control = Y[T == 0]
        self.outcome_model_0 = clone(self.outcome_model)
        self.outcome_model_0.fit(X_control, Y_control)
        
        # 予測値
        self.mu_1_hat = self.outcome_model_1.predict(X)
        self.mu_0_hat = self.outcome_model_0.predict(X)
        
        if self.verbose:
            print(f"   処置群モデル: {len(Y_treated)} サンプルで学習")
            print(f"   統制群モデル: {len(Y_control)} サンプルで学習")
        
        # 3. DR推定量の計算
        if self.verbose:
            print("\n🎯 Step 3: ダブルロバスト推定量の計算")
            print("-"*40)
        
        self.ate_result = self._compute_dr_ate(X, T, Y)
        
        if self.verbose:
            self._print_results()
        
        return self
    
    def _validate_inputs(self, X, T, Y):
        """入力データの検証"""
        if len(X) != len(T) or len(X) != len(Y):
            raise ValueError("X, T, Y must have the same length")
        
        if not np.all(np.isin(T, [0, 1])):
            raise ValueError("T must be binary (0 or 1)")
        
        if np.any(np.isnan(X)) or np.any(np.isnan(T)) or np.any(np.isnan(Y)):
            raise ValueError("Input contains NaN values")
    
    def _compute_dr_ate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> DREstimatorResult:
        """
        ダブルロバスト推定量の計算
        
        理論背景:
        - Robins et al. (1994) のAugmented IPW推定量
        - Bang & Robins (2005) のダブルロバスト推定
        """
        n = len(Y)
        
        # 影響関数の計算
        psi = np.zeros(n)
        
        for i in range(n):
            # 回帰調整項（Outcome regression component）
            regression_term = self.mu_1_hat[i] - self.mu_0_hat[i]
            
            # IPW補正項（Propensity score weighting component）
            if T[i] == 1:
                # 処置群の補正
                ipw_correction = (Y[i] - self.mu_1_hat[i]) / self.e_hat[i]
            else:
                # 統制群の補正
                ipw_correction = -(Y[i] - self.mu_0_hat[i]) / (1 - self.e_hat[i])
            
            # ダブルロバスト推定量の影響関数
            psi[i] = regression_term + ipw_correction
        
        # ATE点推定
        ate = np.mean(psi)
        
        # 標準誤差の計算（サンドイッチ分散推定）
        var_psi = np.var(psi, ddof=1)  # 不偏分散
        se = np.sqrt(var_psi / n)
        
        # 95%信頼区間
        z_alpha = 1.96  # 正規分布の97.5%点
        ci_lower = ate - z_alpha * se
        ci_upper = ate + z_alpha * se
        
        return DREstimatorResult(
            ate=ate,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            influence_function=psi,
            n_treated=np.sum(T),
            n_control=np.sum(1 - T),
            propensity_scores=self.e_hat,
            outcome_predictions_1=self.mu_1_hat,
            outcome_predictions_0=self.mu_0_hat
        )
    
    def _print_results(self):
        """結果の表示"""
        result = self.ate_result
        
        print(f"\n🎯 推定結果:")
        print(f"   ATE = {result.ate:,.2f} 円")
        print(f"   SE  = {result.se:,.2f} 円")
        print(f"   95% CI = [{result.ci_lower:,.2f}, {result.ci_upper:,.2f}]")
        
        # 統計的有意性の判定
        is_significant = result.ci_lower > 0 or result.ci_upper < 0
        if is_significant:
            if result.ate > 0:
                print(f"   ✅ 統計的に有意なポジティブ効果")
            else:
                print(f"   ⚠️ 統計的に有意なネガティブ効果")
        else:
            print(f"   ❌ 統計的に有意ではない")
    
    def get_diagnostics(self) -> Dict:
        """
        診断統計量の計算
        """
        if self.ate_result is None:
            raise ValueError("Model must be fitted first")
        
        # 傾向スコアの分布統計量
        ps_stats = {
            'min': np.min(self.e_hat),
            'max': np.max(self.e_hat),
            'mean': np.mean(self.e_hat),
            'std': np.std(self.e_hat),
            'extreme_low': np.mean(self.e_hat < 0.1),
            'extreme_high': np.mean(self.e_hat > 0.9)
        }
        
        # 有効サンプルサイズ（Effective Sample Size）
        # Kish (1965) の定義に基づく
        weights_treated = 1 / self.e_hat
        weights_control = 1 / (1 - self.e_hat)
        
        # 処置群のESS
        treated_mask = self.ate_result.n_treated
        ess_treated = np.sum(weights_treated)**2 / np.sum(weights_treated**2)
        
        # 統制群のESS
        ess_control = np.sum(weights_control)**2 / np.sum(weights_control**2)
        
        return {
            'propensity_score': ps_stats,
            'ess_treated': ess_treated,
            'ess_control': ess_control,
            'ess_ratio': min(ess_treated, ess_control) / len(self.e_hat),
            'max_weight': max(np.max(weights_treated), np.max(weights_control))
        }
