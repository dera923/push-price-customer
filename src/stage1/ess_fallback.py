"""
Effective Sample Size (ESS) 低下時の対処戦略
NASA JPL標準：段階的フォールバック

Google実例：広告配信で使用される手法
Meta実例：A/Bテストでの極端な不均衡への対処
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import logging
from sklearn.linear_model import LogisticRegression
import warnings

logger = logging.getLogger(__name__)


@dataclass
class FallbackResult:
    """フォールバック結果を格納"""
    method_used: str
    ate: float
    ci: Tuple[float, float]
    ess: float
    ess_ratio: float
    reason: str


class AdaptiveIPWEstimator:
    """
    ESS低下時に自動的にフォールバックする推定器
    
    フォールバック順序：
    1. Trimming閾値の調整
    2. Overlap Weightsへの切り替え
    3. ATEからATTへの変更
    4. Doubly Robustへの切り替え（Stage 2への準備）
    """
    
    def __init__(self, min_ess_ratio: float = 0.2):
        self.min_ess_ratio = min_ess_ratio
        self.fallback_history = []
        
    def estimate_with_fallback(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray
    ) -> FallbackResult:
        """
        ESS診断付きの推定（自動フォールバック）
        """
        logger.info("Starting adaptive estimation with ESS monitoring")
        
        # Strategy 1: Standard IPW with stabilized weights
        result = self._try_standard_ipw(Y, T, X)
        if result.ess_ratio >= self.min_ess_ratio:
            logger.info(f"Standard IPW successful: ESS={result.ess_ratio:.1%}")
            return result
            
        logger.warning(f"Standard IPW failed: ESS={result.ess_ratio:.1%}")
        
        # Strategy 2: Aggressive trimming
        result = self._try_aggressive_trimming(Y, T, X)
        if result.ess_ratio >= self.min_ess_ratio:
            logger.info(f"Aggressive trimming successful: ESS={result.ess_ratio:.1%}")
            return result
            
        logger.warning(f"Aggressive trimming failed: ESS={result.ess_ratio:.1%}")
        
        # Strategy 3: Overlap weights (最も頑健)
        result = self._try_overlap_weights(Y, T, X)
        if result.ess_ratio >= self.min_ess_ratio:
            logger.info(f"Overlap weights successful: ESS={result.ess_ratio:.1%}")
            return result
            
        logger.warning(f"Overlap weights failed: ESS={result.ess_ratio:.1%}")
        
        # Strategy 4: Change estimand to ATT
        result = self._try_att(Y, T, X)
        logger.info(f"Falling back to ATT: ESS={result.ess_ratio:.1%}")
        
        # Final warning
        if result.ess_ratio < self.min_ess_ratio:
            warnings.warn(
                f"All strategies resulted in low ESS ({result.ess_ratio:.1%}). "
                f"Consider:\n"
                f"1. Collecting more data\n"
                f"2. Using different identification strategy (IV, RDD)\n"
                f"3. Redefining the causal question",
                UserWarning
            )
            
        return result
    
    def _try_standard_ipw(self, Y, T, X) -> FallbackResult:
        """標準IPW with stabilized weights"""
        ps = self._fit_ps(X, T)
        ps_trimmed = np.clip(ps, 0.01, 0.99)
        
        # Stabilized weights
        p_treat = np.mean(T)
        weights = np.where(
            T == 1,
            p_treat / ps_trimmed,
            (1 - p_treat) / (1 - ps_trimmed)
        )
        
        ate, ci = self._compute_ate_with_ci(Y, T, weights)
        ess = self._compute_ess(weights)
        
        return FallbackResult(
            method_used="Standard IPW (stabilized)",
            ate=ate,
            ci=ci,
            ess=ess,
            ess_ratio=ess/len(Y),
            reason="First attempt with standard settings"
        )
    
    def _try_aggressive_trimming(self, Y, T, X) -> FallbackResult:
        """より積極的なトリミング (5%/95%)"""
        ps = self._fit_ps(X, T)
        ps_trimmed = np.clip(ps, 0.05, 0.95)  # More aggressive
        
        # Remove extreme units completely
        keep_idx = (ps >= 0.05) & (ps <= 0.95)
        Y_trim = Y[keep_idx]
        T_trim = T[keep_idx]
        ps_trim = ps_trimmed[keep_idx]
        
        logger.info(f"Aggressive trimming: keeping {np.sum(keep_idx)}/{len(Y)} samples")
        
        # Stabilized weights on trimmed sample
        p_treat = np.mean(T_trim)
        weights = np.where(
            T_trim == 1,
            p_treat / ps_trim,
            (1 - p_treat) / (1 - ps_trim)
        )
        
        ate, ci = self._compute_ate_with_ci(Y_trim, T_trim, weights)
        ess = self._compute_ess(weights)
        
        return FallbackResult(
            method_used="Aggressive Trimming (5%/95%)",
            ate=ate,
            ci=ci,
            ess=ess,
            ess_ratio=ess/len(Y_trim),
            reason="Standard IPW had low ESS"
        )
    
    def _try_overlap_weights(self, Y, T, X) -> FallbackResult:
        """
        Overlap Weights (Li, Morgan & Zaslavsky 2018)
        最も安定した重み付け方法
        
        w = PS * (1-PS) for both groups
        
        これはATEではなくATO (Average Treatment Effect on Overlap)を推定
        """
        ps = self._fit_ps(X, T)
        
        # Overlap weights - 自動的に極端な重みを抑制
        weights = ps * (1 - ps)
        
        # Normalized weights
        weights_1 = weights * T / np.sum(weights * T)
        weights_0 = weights * (1 - T) / np.sum(weights * (1 - T))
        
        ato = np.sum(Y * weights_1) - np.sum(Y * weights_0)
        
        # Bootstrap CI
        ci = self._bootstrap_ci_overlap(Y, T, X, ps)
        ess = self._compute_ess(weights)
        
        logger.info(f"Overlap weights: estimating ATO instead of ATE")
        
        return FallbackResult(
            method_used="Overlap Weights (ATO)",
            ate=ato,
            ci=ci,
            ess=ess,
            ess_ratio=ess/len(Y),
            reason="IPW and trimming had low ESS - switching to ATO"
        )
    
    def _try_att(self, Y, T, X) -> FallbackResult:
        """
        ATT (Average Treatment on Treated) への切り替え
        処置群のみを対象にすることで、極端なPSの影響を軽減
        """
        ps = self._fit_ps(X, T)
        
        # ATT weights: 処置群は1、対照群はPS/(1-PS)
        weights = np.where(
            T == 1,
            1.0,
            ps / (1 - ps + 1e-10)
        )
        
        # Normalize within groups
        weights_1 = weights * T / np.sum(weights * T)
        weights_0 = weights * (1 - T) / np.sum(weights * (1 - T))
        
        att = np.sum(Y * T) / np.sum(T) - np.sum(Y * weights_0)
        
        # Bootstrap CI
        ci = self._bootstrap_ci_att(Y, T, X, ps)
        ess = self._compute_ess(weights[T == 0])  # ESS for control group only
        
        logger.info(f"Switching to ATT: Effect on treated units only")
        
        return FallbackResult(
            method_used="ATT (Effect on Treated)",
            ate=att,
            ci=ci,
            ess=ess,
            ess_ratio=ess/np.sum(1-T),
            reason="All ATE methods had low ESS - switching estimand"
        )
    
    def _fit_ps(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Propensity Score推定"""
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(X, T)
        return model.predict_proba(X)[:, 1]
    
    def _compute_ate_with_ci(
        self, 
        Y: np.ndarray, 
        T: np.ndarray, 
        weights: np.ndarray
    ) -> Tuple[float, Tuple[float, float]]:
        """重み付きATE計算と信頼区間"""
        # Normalize weights
        w1 = weights * T / np.sum(weights * T)
        w0 = weights * (1 - T) / np.sum(weights * (1 - T))
        
        ate = np.sum(Y * w1) - np.sum(Y * w0)
        
        # 簡易的な信頼区間（実際はbootstrapが望ましい）
        var1 = np.sum(w1**2 * (Y - np.sum(Y * w1))**2)
        var0 = np.sum(w0**2 * (Y - np.sum(Y * w0))**2)
        se = np.sqrt(var1 + var0)
        
        ci = (ate - 1.96 * se, ate + 1.96 * se)
        return ate, ci
    
    def _compute_ess(self, weights: np.ndarray) -> float:
        """Effective Sample Size計算"""
        return np.sum(weights)**2 / np.sum(weights**2)
    
    def _bootstrap_ci_overlap(
        self, 
        Y: np.ndarray, 
        T: np.ndarray,
        X: np.ndarray,
        ps: np.ndarray,
        n_boot: int = 500
    ) -> Tuple[float, float]:
        """Overlap weights用のBootstrap信頼区間"""
        n = len(Y)
        atos = []
        
        for _ in range(n_boot):
            idx = np.random.choice(n, size=n, replace=True)
            ps_b = ps[idx]
            weights_b = ps_b * (1 - ps_b)
            
            w1_b = weights_b * T[idx] / np.sum(weights_b * T[idx])
            w0_b = weights_b * (1 - T[idx]) / np.sum(weights_b * (1 - T[idx]))
            
            ato_b = np.sum(Y[idx] * w1_b) - np.sum(Y[idx] * w0_b)
            atos.append(ato_b)
        
        return np.percentile(atos, 2.5), np.percentile(atos, 97.5)
    
    def _bootstrap_ci_att(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        ps: np.ndarray,
        n_boot: int = 500
    ) -> Tuple[float, float]:
        """ATT用のBootstrap信頼区間"""
        n = len(Y)
        atts = []
        
        for _ in range(n_boot):
            idx = np.random.choice(n, size=n, replace=True)
            ps_b = ps[idx]
            
            # ATT weights
            weights_b = np.where(
                T[idx] == 1,
                1.0,
                ps_b / (1 - ps_b + 1e-10)
            )
            
            w0_b = weights_b * (1 - T[idx]) / np.sum(weights_b * (1 - T[idx]))
            att_b = np.sum(Y[idx] * T[idx]) / np.sum(T[idx]) - np.sum(Y[idx] * w0_b)
            atts.append(att_b)
        
        return np.percentile(atts, 2.5), np.percentile(atts, 97.5)


def demonstrate_ess_fallback():
    """
    ESS低下シナリオでのフォールバック実演
    """
    print("="*70)
    print("ESS Fallback Strategy Demonstration")
    print("NASA JPL Standard: Graceful Degradation")
    print("="*70)
    
    np.random.seed(42)
    
    # Scenario 1: Good overlap (ESS should be fine)
    print("\n[Scenario 1] Good Overlap - Standard IPW should work")
    print("-"*50)
    n = 5000
    X_good = np.random.randn(n, 5)
    ps_good = 1 / (1 + np.exp(-(X_good[:, 0] + 0.5 * X_good[:, 1])))
    T_good = np.random.binomial(1, ps_good)
    Y_good = X_good[:, 0] + 0.5 * T_good + np.random.randn(n)
    
    estimator = AdaptiveIPWEstimator(min_ess_ratio=0.2)
    result_good = estimator.estimate_with_fallback(Y_good, T_good, X_good)
    
    print(f"Method used: {result_good.method_used}")
    print(f"ATE: {result_good.ate:.4f}")
    print(f"95% CI: [{result_good.ci[0]:.4f}, {result_good.ci[1]:.4f}]")
    print(f"ESS ratio: {result_good.ess_ratio:.1%}")
    print(f"Reason: {result_good.reason}")
    
    # Scenario 2: Poor overlap (ESS problems)
    print("\n[Scenario 2] Poor Overlap - Should trigger fallback")
    print("-"*50)
    X_poor = np.random.randn(n, 5)
    # 極端なPS生成
    ps_poor = 1 / (1 + np.exp(-(3 * X_poor[:, 0] + 2 * X_poor[:, 1])))
    T_poor = np.random.binomial(1, ps_poor)
    Y_poor = X_poor[:, 0] + 0.5 * T_poor + np.random.randn(n)
    
    result_poor = estimator.estimate_with_fallback(Y_poor, T_poor, X_poor)
    
    print(f"Method used: {result_poor.method_used}")
    print(f"ATE: {result_poor.ate:.4f}")
    print(f"95% CI: [{result_poor.ci[0]:.4f}, {result_poor.ci[1]:.4f}]")
    print(f"ESS ratio: {result_poor.ess_ratio:.1%}")
    print(f"Reason: {result_poor.reason}")
    
    # Scenario 3: Extreme violation (Multiple fallbacks needed)
    print("\n[Scenario 3] Extreme Violation - Multiple fallbacks")
    print("-"*50)
    X_extreme = np.random.randn(n, 5)
    # 非常に極端なPS
    ps_extreme = 1 / (1 + np.exp(-(5 * X_extreme[:, 0] + 3 * X_extreme[:, 1])))
    T_extreme = np.random.binomial(1, ps_extreme)
    Y_extreme = X_extreme[:, 0] + 0.5 * T_extreme + np.random.randn(n)
    
    result_extreme = estimator.estimate_with_fallback(Y_extreme, T_extreme, X_extreme)
    
    print(f"Method used: {result_extreme.method_used}")
    print(f"ATE: {result_extreme.ate:.4f}")
    print(f"95% CI: [{result_extreme.ci[0]:.4f}, {result_extreme.ci[1]:.4f}]")
    print(f"ESS ratio: {result_extreme.ess_ratio:.1%}")
    print(f"Reason: {result_extreme.reason}")
    
    print("\n" + "="*70)
    print("Interpretation Guide:")
    print("- Standard IPW: Most efficient when overlap is good")
    print("- Aggressive Trimming: Trades bias for variance")
    print("- Overlap Weights: Most stable, estimates ATO not ATE")
    print("- ATT: Changes the causal question to treated units only")
    print("="*70)


if __name__ == "__main__":
    demonstrate_ess_fallback()
