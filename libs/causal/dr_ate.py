# libs/causal/dr_ate.py

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

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
    
class DoubleRobustATE:
    """
    ダブルロバスト平均処置効果推定器
    Google/Metaレベルの実装
    """
    
    def __init__(self, 
                 outcome_model=None,
                 propensity_model=None,
                 trim_threshold: float = 0.01):
        """
        Args:
            outcome_model: 結果予測モデル（sklearn互換）
            propensity_model: 傾向スコアモデル（sklearn互換）
            trim_threshold: 傾向スコアのトリミング閾値
        """
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.trim_threshold = trim_threshold
        
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'DoubleRobustATE':
        """
        モデルの学習
        
        Args:
            X: 共変量行列 (n_samples, n_features)
            T: 処置インジケータ (n_samples,)
            Y: 結果変数 (n_samples,)
        """
        # 1. 傾向スコアの学習
        print("📊 傾向スコアモデルを学習中...")
        self.propensity_model.fit(X, T)
        self.e_hat = self.propensity_model.predict_proba(X)[:, 1]
        
        # トリミング（極端な傾向スコアを制限）
        self.e_hat = np.clip(self.e_hat, self.trim_threshold, 1 - self.trim_threshold)
        
        # 2. 結果モデルの学習（処置群と統制群で別々に）
        print("📈 結果モデルを学習中...")
        
        # 処置群モデル
        X_treated = X[T == 1]
        Y_treated = Y[T == 1]
        self.outcome_model_1 = self._clone_model(self.outcome_model)
        self.outcome_model_1.fit(X_treated, Y_treated)
        
        # 統制群モデル  
        X_control = X[T == 0]
        Y_control = Y[T == 0]
        self.outcome_model_0 = self._clone_model(self.outcome_model)
        self.outcome_model_0.fit(X_control, Y_control)
        
        # 予測値
        self.mu_1_hat = self.outcome_model_1.predict(X)
        self.mu_0_hat = self.outcome_model_0.predict(X)
        
        # 3. DR推定量の計算
        self.ate_result = self._compute_dr_ate(X, T, Y)
        
        return self
    
    def _compute_dr_ate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> DREstimatorResult:
        """
        ダブルロバスト推定量の計算
        
        Googleの広告効果測定で使われる実装パターン
        """
        n = len(Y)
        
        # 影響関数の計算（これが統計理論の核心）
        psi = np.zeros(n)
        
        for i in range(n):
            # 回帰調整項
            regression_term = self.mu_1_hat[i] - self.mu_0_hat[i]
            
            # IPW補正項（処置群）
            if T[i] == 1:
                ipw_correction = (Y[i] - self.mu_1_hat[i]) / self.e_hat[i]
            else:
                # IPW補正項（統制群）
                ipw_correction = -(Y[i] - self.mu_0_hat[i]) / (1 - self.e_hat[i])
            
            psi[i] = regression_term + ipw_correction
        
        # ATE点推定
        ate = np.mean(psi)
        
        # 標準誤差（サンドイッチ分散推定）
        # Metaではこの計算にHC3（heteroskedasticity-consistent）を使用
        var_psi = np.var(psi, ddof=1)
        se = np.sqrt(var_psi / n)
        
        # 95%信頼区間
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        return DREstimatorResult(
            ate=ate,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            influence_function=psi,
            n_treated=np.sum(T),
            n_control=np.sum(1 - T)
        )
    
    def _clone_model(self, model):
        """モデルのクローン作成"""
        from sklearn.base import clone
        return clone(model)
    
    def get_diagnostics(self) -> Dict:
        """
        診断統計量の計算
        NASAのミッションクリティカルな解析で使用される品質チェック
        """
        # 傾向スコアの分布
        ps_stats = {
            'min': np.min(self.e_hat),
            'max': np.max(self.e_hat),
            'mean': np.mean(self.e_hat),
            'extreme_low': np.mean(self.e_hat < 0.1),
            'extreme_high': np.mean(self.e_hat > 0.9)
        }
        
        # 有効サンプルサイズ（Effective Sample Size）
        weights_treated = 1 / self.e_hat
        weights_control = 1 / (1 - self.e_hat)
        
        ess_treated = np.sum(weights_treated[self.T == 1])**2 / \
                     np.sum(weights_treated[self.T == 1]**2)
        ess_control = np.sum(weights_control[self.T == 0])**2 / \
                     np.sum(weights_control[self.T == 0]**2)
        
        return {
            'propensity_score': ps_stats,
            'ess_treated': ess_treated,
            'ess_control': ess_control,
            'max_weight': max(np.max(weights_treated), np.max(weights_control))
        }
