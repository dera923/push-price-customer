"""
ダブルロバスト（DR）ATE推定器の完全自作実装
Google/Meta/NASAレベルの厳密性と数値安定性

理論的基盤：
- Robins et al. (1994): Doubly Robust Estimation
- Bang & Robins (2005): Doubly Robust Estimators
- Kennedy (2023): Towards optimal doubly robust estimation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Tuple, Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class DRATEEstimator:
    """
    ダブルロバスト平均処置効果（ATE）推定器
    
    数学的定式化：
    τ̂_DR = (1/n) Σᵢ [μ̂₁(Xᵢ) - μ̂₀(Xᵢ) + 
                    Tᵢ{Yᵢ - μ̂₁(Xᵢ)}/ê(Xᵢ) - 
                    (1-Tᵢ){Yᵢ - μ̂₀(Xᵢ)}/(1-ê(Xᵢ))]
    
    特徴：
    - ダブルロバスト性（2つのモデルのうち1つが正しければ一致推定）
    - 影響関数による標準誤差計算
    - クロスフィット対応（過学習対策）
    - 数値安定性の確保
    """
    
    def __init__(self, 
                 outcome_model: str = 'random_forest',
                 propensity_model: str = 'logistic',
                 cross_fit: bool = True,
                 n_folds: int = 5,
                 trim_threshold: float = 0.05,
                 random_state: int = 42):
        """
        Parameters:
        -----------
        outcome_model : str
            アウトカム回帰モデル ('random_forest', 'ridge')
        propensity_model : str  
            傾向スコアモデル ('logistic', 'random_forest')
        cross_fit : bool
            クロスフィット使用フラグ
        n_folds : int
            クロスフィットの分割数
        trim_threshold : float
            傾向スコアの裾切り閾値（数値安定性のため）
        """
        
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.cross_fit = cross_fit
        self.n_folds = n_folds
        self.trim_threshold = trim_threshold
        self.random_state = random_state
        
        # 推定結果保存用
        self.ate_ = None
        self.se_ = None
        self.ci_ = None
        self.influence_function_ = None
        self.diagnostics_ = {}
        
    def _get_outcome_model(self) -> Union[RandomForestRegressor, Ridge]:
        """アウトカム回帰モデルの取得"""
        if self.outcome_model == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=20,
                random_state=self.random_state
            )
        elif self.outcome_model == 'ridge':
            return Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown outcome model: {self.outcome_model}")
            
    def _get_propensity_model(self) -> Union[LogisticRegression, RandomForestClassifier]:
        """傾向スコアモデルの取得"""
        if self.propensity_model == 'logistic':
            return LogisticRegression(
                penalty='l2',
                C=1.0,
                max_iter=1000,
                random_state=self.random_state
            )
        elif self.propensity_model == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=20,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown propensity model: {self.propensity_model}")
    
    def _fit_models_cross_fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        クロスフィットによるモデル学習
        過学習を防ぎ、Neyman直交性を保証
        """
        n = X.shape[0]
        
        # 予測値初期化
        mu1_pred = np.zeros(n)
        mu0_pred = np.zeros(n) 
        ps_pred = np.zeros(n)
        
        # K-fold分割
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            # アウトカム回帰（処置群）
            treated_idx = T_train == 1
            if np.sum(treated_idx) > 10:  # 最小サンプルサイズチェック
                mu1_model = self._get_outcome_model()
                mu1_model.fit(X_train[treated_idx], Y_train[treated_idx])
                mu1_pred[test_idx] = mu1_model.predict(X_test)
            
            # アウトカム回帰（対照群）  
            control_idx = T_train == 0
            if np.sum(control_idx) > 10:
                mu0_model = self._get_outcome_model()
                mu0_model.fit(X_train[control_idx], Y_train[control_idx])
                mu0_pred[test_idx] = mu0_model.predict(X_test)
                
            # 傾向スコア
            ps_model = self._get_propensity_model()
            ps_model.fit(X_train, T_train)
            ps_pred[test_idx] = ps_model.predict_proba(X_test)[:, 1]
            
        return mu1_pred, mu0_pred, ps_pred
    
    def _fit_models_standard(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        標準的なモデル学習（非クロスフィット）
        """
        # アウトカム回帰（処置群）
        treated_idx = T == 1
        mu1_model = self._get_outcome_model()
        mu1_model.fit(X[treated_idx], Y[treated_idx])
        mu1_pred = mu1_model.predict(X)
        
        # アウトカム回帰（対照群）
        control_idx = T == 0  
        mu0_model = self._get_outcome_model()
        mu0_model.fit(X[control_idx], Y[control_idx])
        mu0_pred = mu0_model.predict(X)
        
        # 傾向スコア
        ps_model = self._get_propensity_model()
        ps_model.fit(X, T)
        ps_pred = ps_model.predict_proba(X)[:, 1]
        
        return mu1_pred, mu0_pred, ps_pred
    
    def _trim_propensity_scores(self, ps: np.ndarray) -> np.ndarray:
        """
        傾向スコアの裾切り（数値安定性のため）
        極端な値を除去し、重みの爆発を防ぐ
        """
        ps_trimmed = np.clip(ps, self.trim_threshold, 1 - self.trim_threshold)
        
        # 診断情報保存
        n_trimmed = np.sum((ps < self.trim_threshold) | (ps > 1 - self.trim_threshold))
        self.diagnostics_['n_trimmed_ps'] = n_trimmed
        self.diagnostics_['trim_rate'] = n_trimmed / len(ps)
        
        return ps_trimmed
    
    def _calculate_dr_ate(self, 
                         X: np.ndarray, 
                         T: np.ndarray, 
                         Y: np.ndarray,
                         mu1: np.ndarray,
                         mu0: np.ndarray, 
                         ps: np.ndarray) -> float:
        """
        DR-ATE推定値の計算
        
        数式：
        τ̂ = (1/n) Σᵢ [μ₁(Xᵢ) - μ₀(Xᵢ) + 
                      Tᵢ{Yᵢ - μ₁(Xᵢ)}/e(Xᵢ) - 
                      (1-Tᵢ){Yᵢ - μ₀(Xᵢ)}/(1-e(Xᵢ))]
        """
        n = len(Y)
        
        # 各成分の計算
        outcome_diff = mu1 - mu0  # μ₁(X) - μ₀(X)
        
        treated_adjustment = T * (Y - mu1) / ps  # T{Y-μ₁}/e
        control_adjustment = (1 - T) * (Y - mu0) / (1 - ps)  # (1-T){Y-μ₀}/(1-e)
        
        # DR推定値
        dr_components = outcome_diff + treated_adjustment - control_adjustment
        ate = np.mean(dr_components)
        
        # 影響関数（標準誤差計算用）
        self.influence_function_ = dr_components - ate
        
        return ate
    
    def _calculate_standard_error(self) -> float:
        """
        影響関数を用いた標準誤差計算
        
        SE(τ̂) = √(Var(ψᵢ)/n) = √((1/n²)Σᵢψᵢ²)
        ここで ψᵢ は影響関数
        """
        if self.influence_function_ is None:
            raise ValueError("影響関数が計算されていません")
            
        n = len(self.influence_function_)
        variance = np.var(self.influence_function_, ddof=1)
        se = np.sqrt(variance / n)
        
        return se
    
    def _calculate_confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """
        信頼区間の計算
        正規近似を使用
        """
        if self.ate_ is None or self.se_ is None:
            raise ValueError("ATE推定値または標準誤差が計算されていません")
            
        z_critical = stats.norm.ppf(1 - alpha/2)
        lower = self.ate_ - z_critical * self.se_
        upper = self.ate_ + z_critical * self.se_
        
        return lower, upper
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'DRATEEstimator':
        """
        DR-ATE推定器の学習
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            共変量行列
        T : array-like, shape (n_samples,)
            処置変数（0 or 1）
        Y : array-like, shape (n_samples,)
            結果変数
        """
        
        # 入力検証
        X = np.asarray(X)
        T = np.asarray(T)
        Y = np.asarray(Y)
        
        if not np.all(np.isin(T, [0, 1])):
            raise ValueError("処置変数Tは0または1である必要があります")
            
        n, p = X.shape
        
        # 基本統計
        self.diagnostics_['n_samples'] = n
        self.diagnostics_['n_features'] = p
        self.diagnostics_['treatment_rate'] = np.mean(T)
        
        # モデル学習
        if self.cross_fit:
            mu1_pred, mu0_pred, ps_pred = self._fit_models_cross_fit(X, T, Y)
        else:
            mu1_pred, mu0_pred, ps_pred = self._fit_models_standard(X, T, Y)
            
        # 傾向スコアの裾切り
        ps_trimmed = self._trim_propensity_scores(ps_pred)
        
        # DR-ATE推定
        self.ate_ = self._calculate_dr_ate(X, T, Y, mu1_pred, mu0_pred, ps_trimmed)
        
        # 標準誤差計算
        self.se_ = self._calculate_standard_error()
        
        # 信頼区間計算
        self.ci_ = self._calculate_confidence_interval()
        
        # 診断統計の保存
        self.diagnostics_['outcome_mse_treated'] = np.mean((Y[T==1] - mu1_pred[T==1])**2)
        self.diagnostics_['outcome_mse_control'] = np.mean((Y[T==0] - mu0_pred[T==0])**2)
        self.diagnostics_['ps_mean'] = np.mean(ps_pred)
        self.diagnostics_['ps_std'] = np.std(ps_pred)
        
        return self
    
    def ate(self) -> float:
        """ATE推定値の取得"""
        if self.ate_ is None:
            raise ValueError("まずfit()を実行してください")
        return self.ate_
    
    def standard_error(self) -> float:
        """標準誤差の取得"""  
        if self.se_ is None:
            raise ValueError("まずfit()を実行してください")
        return self.se_
        
    def confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """信頼区間の取得"""
        if self.ci_ is None:
            raise ValueError("まずfit()を実行してください")
        return self.ci_
    
    def p_value(self) -> float:
        """p値の計算（帰無仮説：ATE=0）"""
        if self.ate_ is None or self.se_ is None:
            raise ValueError("まずfit()を実行してください")
            
        t_stat = self.ate_ / self.se_
        p_val = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
        
        return p_val
    
    def summary(self) -> Dict:
        """推定結果のサマリー"""
        if self.ate_ is None:
            raise ValueError("まずfit()を実行してください")
            
        ci_lower, ci_upper = self.ci_
        
        return {
            'ate': self.ate_,
            'se': self.se_,
            'p_value': self.p_value(),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': self.p_value() < 0.05,
            'diagnostics': self.diagnostics_
        }

# 使用例とテスト
def test_dr_ate_estimator():
    """DR-ATE推定器のテスト"""
    
    # サンプルデータ生成
    np.random.seed(42)
    n = 1000
    
    # 共変量
    X = np.random.normal(0, 1, (n, 5))
    
    # 真の傾向スコア（ロジスティック）
    ps_true = 1 / (1 + np.exp(-(0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2])))
    T = np.random.binomial(1, ps_true)
    
    # 真のアウトカム（処置効果=5）
    true_ate = 5.0
    Y0 = 2 + 0.8 * X[:, 0] - 0.5 * X[:, 1] + np.random.normal(0, 1, n)
    Y1 = Y0 + true_ate
    Y = T * Y1 + (1 - T) * Y0
    
    # DR-ATE推定
    estimator = DRATEEstimator(cross_fit=True)
    estimator.fit(X, T, Y)
    
    # 結果表示
    results = estimator.summary()
    print("=== DR-ATE推定結果 ===")
    print(f"真のATE: {true_ate:.3f}")
    print(f"推定ATE: {results['ate']:.3f}")
    print(f"標準誤差: {results['se']:.3f}")
    print(f"95%信頼区間: [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]")
    print(f"p値: {results['p_value']:.6f}")
    print(f"有意性: {'有意' if results['significant'] else '非有意'}")
    
    return estimator, results

if __name__ == "__main__":
    estimator, results = test_dr_ate_estimator()
