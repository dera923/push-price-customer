"""
Unified Dimension Reduction Analyzer
==================================

SVD・PCA・NMFの統合比較分析システム
Google/Meta/NASAレベルの包括的評価フレームワーク

核心価値：
1. 理論的完璧性：各手法の数学的特性を完全理解
2. 実用性：ビジネス課題に最適な手法を自動選択
3. 解釈可能性：結果の意味を明確に説明
4. スケーラビリティ：大規模データに対応

評価観点：
- 数値精度（復元誤差・収束性）
- 計算効率（時間・メモリ）
- 解釈可能性（因子の意味）
- 汎化性能（新データへの適用）
- 業務適用性（推薦・セグメント等）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import time
import warnings
from pathlib import Path
import json

# 自作モジュールのインポート
from src.customer.dimension_reduction.algorithms.svd.svd_decomposer import SVDDecomposer, SVDConfig
from src.customer.dimension_reduction.algorithms.pca.pca_analyzer import PCAAnalyzer, PCAConfig
from src.customer.dimension_reduction.algorithms.nmf.nmf_factorizer import NMFFactorizer, NMFConfig

logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """統合分析の設定クラス"""
    n_components: int = 10
    test_size_ratio: float = 0.2  # テストデータの割合
    cross_validation_folds: int = 5
    evaluation_metrics: List[str] = None
    save_results: bool = True
    output_dir: str = "results/dimension_reduction"
    random_state: int = 42
    
    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = [
                "reconstruction_error",
                "explained_variance",
                "sparsity_ratio",
                "computation_time",
                "memory_usage",
                "interpretability_score"
            ]


class DimensionReductionComparator:
    """
    次元削減手法の包括的比較分析クラス
    
    実装哲学：
    - 公平な比較：同一データ・同一評価基準
    - 多角的評価：精度・効率・解釈性を総合判定
    - 理論裏付け：数学的根拠に基づく結果解釈
    - 実務直結：ビジネス課題解決への具体的示唆
    """
    
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.results_: Dict[str, Any] = {}
        self.models_: Dict[str, Any] = {}
        
        # 出力ディレクトリの作成
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("DimensionReductionComparator初期化完了")
    
    def comprehensive_analysis(self, X: np.ndarray, 
                             feature_names: Optional[List[str]] = None,
                             sample_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        包括的次元削減分析の実行
        
        Args:
            X: 入力データ (n_samples, n_features)
            feature_names: 特徴量名リスト
            sample_names: サンプル名リスト
            
        Returns:
            分析結果の包括的辞書
        """
        logger.info("=== 包括的次元削減分析開始 ===")
        logger.info(f"データサイズ: {X.shape}")
        
        # データ分割
        X_train, X_test = self._split_data(X)
        
        # 各手法の実行と評価
        methods_config = self._get_methods_config()
        
        for method_name, (model_class, config) in methods_config.items():
            logger.info(f"\n--- {method_name}手法の分析開始 ---")
            
            try:
                # モデル学習
                start_time = time.time()
                model = model_class(config)
                
                # メモリ使用量の測定
                import psutil
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024**2  # MB
                
                # フィット実行
                model.fit(X_train)
                
                memory_after = process.memory_info().rss / 1024**2  # MB
                fit_time = time.time() - start_time
                
                # 変換実行
                start_time = time.time()
                X_train_transformed = model.transform(X_train)
                X_test_transformed = model.transform(X_test)
                transform_time = time.time() - start_time
                
                # 結果保存
                self.models_[method_name] = model
                
                # 評価実行
                evaluation = self._evaluate_method(
                    method_name, model, X_train, X_test,
                    X_train_transformed, X_test_transformed,
                    fit_time, transform_time, memory_after - memory_before
                )
                
                self.results_[method_name] = evaluation
                
                logger.info(f"{method_name}完了: フィット{fit_time:.2f}秒, "
                           f"変換{transform_time:.2f}秒")
                
            except Exception as e:
                logger.error(f"{method_name}でエラー: {e}")
                self.results_[method_name] = {"error": str(e)}
        
        # 比較分析
        comparison_results = self._compare_methods()
        
        # 推奨手法の決定
        recommendation = self._recommend_method(X)
        
        # 包括的結果の構築
        comprehensive_results = {
            "data_info": {
                "shape": X.shape,
                "sparsity": float((X == 0).mean()),
                "condition_number": np.linalg.cond(X) if min(X.shape) <= 1000 else None,
                "feature_names": feature_names,
                "sample_names": sample_names
            },
            "individual_results": self.results_,
            "comparison": comparison_results,
            "recommendation": recommendation,
            "analysis_config": self.config.__dict__
        }
        
        # 結果保存
        if self.config.save_results:
            self._save_results(comprehensive_results)
        
        logger.info("=== 包括的分析完了 ===")
        return comprehensive_results
    
    def _split_data(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """データをトレーニング・テストに分割"""
        n_samples = X.shape[0]
        n_train = int(n_samples * (1 - self.config.test_size_ratio))
        
        # 再現性のためのシード設定
        np.random.seed(self.config.random_state)
        indices = np.random.permutation(n_samples)
        
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        return X[train_indices], X[test_indices]
    
    def _get_methods_config(self) -> Dict[str, Tuple]:
        """各手法の設定を取得"""
        return {
            "SVD": (SVDDecomposer, SVDConfig(
                n_components=self.config.n_components,
                algorithm="auto",
                random_state=self.config.random_state
            )),
            "PCA": (PCAAnalyzer, PCAConfig(
                n_components=self.config.n_components,
                algorithm="auto",
                random_state=self.config.random_state
            )),
            "NMF": (NMFFactorizer, NMFConfig(
                n_components=self.config.n_components,
                algorithm="auto",
                max_iter=200,
                random_state=self.config.random_state
            ))
        }
    
    def _evaluate_method(self, method_name: str, model: Any,
                        X_train: np.ndarray, X_test: np.ndarray,
                        X_train_transformed: np.ndarray, X_test_transformed: np.ndarray,
                        fit_time: float, transform_time: float, memory_usage: float) -> Dict[str, Any]:
        """個別手法の包括的評価"""
        
        evaluation = {
            "performance_metrics": {},
            "efficiency_metrics": {
                "fit_time": fit_time,
                "transform_time": transform_time,
                "memory_usage_mb": memory_usage
            },
            "quality_metrics": {},
            "interpretability_metrics": {}
        }
        
        # 1. 性能評価（復元精度）
        if hasattr(model, 'inverse_transform'):
            try:
                X_train_reconstructed = model.inverse_transform(X_train_transformed)
                X_test_reconstructed = model.inverse_transform(X_test_transformed)
                
                train_reconstruction_error = np.mean((X_train - X_train_reconstructed) ** 2)
                test_reconstruction_error = np.mean((X_test - X_test_reconstructed) ** 2)
                
                evaluation["performance_metrics"].update({
                    "train_reconstruction_error": float(train_reconstruction_error),
                    "test_reconstruction_error": float(test_reconstruction_error),
                    "generalization_gap": float(test_reconstruction_error - train_reconstruction_error)
                })
            except Exception as e:
                logger.warning(f"{method_name}の復元評価でエラー: {e}")
        
        # 2. 分散説明力（PCA/SVD）
        if hasattr(model, 'explained_variance_ratio_'):
            explained_variance = model.explained_variance_ratio_
            evaluation["performance_metrics"].update({
                "explained_variance_ratio": explained_variance.tolist(),
                "cumulative_variance_ratio": np.cumsum(explained_variance).tolist(),
                "total_variance_explained": float(np.sum(explained_variance))
            })
        
        # 3. スパース性評価
        sparsity_original = (X_train == 0).mean()
        sparsity_transformed = (X_train_transformed == 0).mean()
        
        evaluation["quality_metrics"].update({
            "original_sparsity": float(sparsity_original),
            "transformed_sparsity": float(sparsity_transformed),
            "sparsity_preservation": float(sparsity_transformed / sparsity_original) if sparsity_original > 0 else 0.0
        })
        
        # 4. 解釈可能性評価（手法別）
        if method_name == "NMF" and hasattr(model, 'get_factor_interpretation'):
            try:
                interpretation = model.get_factor_interpretation(top_k=5)
                evaluation["interpretability_metrics"] = {
                    "factor_importance": interpretation["factor_importance"].tolist(),
                    "factor_sparsity": interpretation["factor_sparsity"].tolist(),
                    "interpretability_score": float(np.mean(interpretation["factor_sparsity"]))
                }
            except Exception as e:
                logger.warning(f"NMF解釈性評価でエラー: {e}")
        
        # 5. 数値安定性評価
        if hasattr(model, 's_'):  # SVD系
            singular_values = model.s_
            condition_number = singular_values[0] / singular_values[-1] if singular_values[-1] > 0 else np.inf
            evaluation["quality_metrics"]["condition_number"] = float(condition_number)
        
        # 6. 変換品質
        transformation_variance = np.var(X_train_transformed, axis=0)
        evaluation["quality_metrics"].update({
            "component_variance_balance": float(np.std(transformation_variance)),
            "min_component_variance": float(np.min(transformation_variance)),
            "max_component_variance": float(np.max(transformation_variance))
        })
        
        return evaluation
    
    def _compare_methods(self) -> Dict[str, Any]:
        """手法間の比較分析"""
        comparison = {
            "performance_ranking": {},
            "efficiency_ranking": {},
            "trade_off_analysis": {},
            "use_case_recommendations": {}
        }
        
        # 有効な結果のみを抽出
        valid_results = {k: v for k, v in self.results_.items() if "error" not in v}
        
        if not valid_results:
            return {"error": "比較可能な結果がありません"}
        
        # 1. 性能ランキング
        performance_metrics = ["test_reconstruction_error", "total_variance_explained"]
        for metric in performance_metrics:
            metric_values = {}
            for method, results in valid_results.items():
                if metric in results.get("performance_metrics", {}):
                    metric_values[method] = results["performance_metrics"][metric]
            
            if metric_values:
                if "error" in metric:  # 誤差系は小さい方が良い
                    sorted_methods = sorted(metric_values.items(), key=lambda x: x[1])
                else:  # 分散説明力は大きい方が良い
                    sorted_methods = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                
                comparison["performance_ranking"][metric] = [
                    {"method": method, "value": float(value)} for method, value in sorted_methods
                ]
        
        # 2. 効率性ランキング
        efficiency_metrics = ["fit_time", "transform_time", "memory_usage_mb"]
        for metric in efficiency_metrics:
            metric_values = {}
            for method, results in valid_results.items():
                if metric in results.get("efficiency_metrics", {}):
                    metric_values[method] = results["efficiency_metrics"][metric]
            
            if metric_values:
                sorted_methods = sorted(metric_values.items(), key=lambda x: x[1])  # 小さい方が良い
                comparison["efficiency_ranking"][metric] = [
                    {"method": method, "value": float(value)} for method, value in sorted_methods
                ]
        
        # 3. トレードオフ分析
        comparison["trade_off_analysis"] = self._analyze_trade_offs(valid_results)
        
        # 4. 用途別推奨
        comparison["use_case_recommendations"] = {
            "highest_accuracy": self._get_best_method(valid_results, "test_reconstruction_error", minimize=True),
            "fastest_computation": self._get_best_method(valid_results, "fit_time", minimize=True),
            "memory_efficient": self._get_best_method(valid_results, "memory_usage_mb", minimize=True),
            "interpretable": self._get_best_method(valid_results, "interpretability_score", minimize=False)
        }
        
        return comparison
    
    def _analyze_trade_offs(self, valid_results: Dict[str, Any]) -> Dict[str, Any]:
        """精度・効率・解釈性のトレードオフ分析"""
        trade_offs = {}
        
        for method, results in valid_results.items():
            # スコア正規化（0-1スケール）
            accuracy_score = 1.0  # デフォルト
            efficiency_score = 1.0
            interpretability_score = 0.0
            
            # 精度スコア（復元誤差の逆数）
            if "test_reconstruction_error" in results.get("performance_metrics", {}):
                error = results["performance_metrics"]["test_reconstruction_error"]
                accuracy_score = 1.0 / (1.0 + error) if error > 0 else 1.0
            
            # 効率スコア（計算時間の逆数）
            if "fit_time" in results.get("efficiency_metrics", {}):
                time_cost = results["efficiency_metrics"]["fit_time"]
                efficiency_score = 1.0 / (1.0 + time_cost) if time_cost > 0 else 1.0
            
            # 解釈性スコア
            if "interpretability_score" in results.get("interpretability_metrics", {}):
                interpretability_score = results["interpretability_metrics"]["interpretability_score"]
            
            # 総合スコア（重み付き平均）
            weights = {"accuracy": 0.4, "efficiency": 0.3, "interpretability": 0.3}
            overall_score = (weights["accuracy"] * accuracy_score + 
                           weights["efficiency"] * efficiency_score + 
                           weights["interpretability"] * interpretability_score)
            
            trade_offs[method] = {
                "accuracy_score": float(accuracy_score),
                "efficiency_score": float(efficiency_score),
                "interpretability_score": float(interpretability_score),
                "overall_score": float(overall_score)
            }
        
        return trade_offs
    
    def _get_best_method(self, valid_results: Dict[str, Any], metric: str, minimize: bool = True) -> Optional[str]:
        """指定メトリックでの最良手法を取得"""
        metric_values = {}
        
        for method, results in valid_results.items():
            # メトリックの場所を探索
            value = None
            for category in ["performance_metrics", "efficiency_metrics", "interpretability_metrics"]:
                if metric in results.get(category, {}):
                    value = results[category][metric]
                    break
            
            if value is not None:
                metric_values[method] = value
        
        if not metric_values:
            return None
        
        if minimize:
            return min(metric_values.items(), key=lambda x: x[1])[0]
        else:
            return max(metric_values.items(), key=lambda x: x[1])[0]
    
    def _recommend_method(self, X: np.ndarray) -> Dict[str, Any]:
        """データ特性に基づく推奨手法の決定"""
        n_samples, n_features = X.shape
        sparsity = (X == 0).mean()
        has_negative = (X < 0).any()
        
        recommendations = []
        
        # データ特性ベースの推奨ロジック
        if not has_negative and sparsity > 0.7:
            recommendations.append({
                "method": "NMF",
                "reason": "非負データで高スパース（>70%）のため、解釈可能な因子抽出に最適",
                "confidence": 0.9
            })
        
        if n_samples > 10000 and n_features > 1000:
            recommendations.append({
                "method": "SVD",
                "reason": "大規模データで数値安定性が重要。ランダム化SVDで高速処理可能",
                "confidence": 0.8
            })
        
        if n_features < n_samples and n_features < 5000:
            recommendations.append({
                "method": "PCA",
                "reason": "中規模データで分散解釈が重要。理論的に最適な次元削減",
                "confidence": 0.85
            })
        
        # 性能結果ベースの追加推奨
        if hasattr(self, 'results_') and self.results_:
            trade_offs = self._analyze_trade_offs({k: v for k, v in self.results_.items() if "error" not in v})
            if trade_offs:
                best_overall = max(trade_offs.items(), key=lambda x: x[1]["overall_score"])
                recommendations.append({
                    "method": best_overall[0],
                    "reason": f"実測結果で総合スコア最高（{best_overall[1]['overall_score']:.3f}）",
                    "confidence": 0.95
                })
        
        return {
            "primary_recommendation": recommendations[0] if recommendations else None,
            "all_recommendations": recommendations,
            "data_characteristics": {
                "size": f"{n_samples:,} x {n_features:,}",
                "sparsity": f"{sparsity:.1%}",
                "has_negative_values": has_negative,
                "size_category": self._categorize_data_size(n_samples, n_features)
            }
        }
    
    def _categorize_data_size(self, n_samples: int, n_features: int) -> str:
        """データサイズのカテゴリ化"""
        total_elements = n_samples * n_features
        
        if total_elements < 1e4:
            return "small"
        elif total_elements < 1e6:
            return "medium"
        elif total_elements < 1e8:
            return "large"
        else:
            return "very_large"
    
    def _save_results(self, results: Dict[str, Any]):
        """分析結果の保存"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # JSON形式で結果保存
        results_file = Path(self.config.output_dir) / f"dimension_reduction_analysis_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"分析結果を保存: {results_file}")
        
        # サマリーレポート生成
        self._generate_summary_report(results, timestamp)
    
    def _generate_summary_report(self, results: Dict[str, Any], timestamp: str):
        """サマリーレポートの生成"""
        report_file = Path(self.config.output_dir) / f"analysis_summary_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 次元削減分析レポート\n\n")
            f.write(f"**分析日時**: {timestamp}\n\n")
            
            # データ情報
            data_info = results.get("data_info", {})
            f.write("## データ概要\n")
            f.write(f"- サイズ: {data_info.get('shape', 'N/A')}\n")
            f.write(f"- スパース率: {data_info.get('sparsity', 0):.1%}\n\n")
            
            # 推奨結果
            recommendation = results.get("recommendation", {})
            primary_rec = recommendation.get("primary_recommendation")
            if primary_rec:
                f.write("## 推奨手法\n")
                f.write(f"**{primary_rec['method']}** (信頼度: {primary_rec['confidence']:.1%})\n\n")
                f.write(f"理由: {primary_rec['reason']}\n\n")
            
            # 比較結果
            comparison = results.get("comparison", {})
            if "performance_ranking" in comparison:
                f.write("## 性能ランキング\n")
                for metric, ranking in comparison["performance_ranking"].items():
                    f.write(f"### {metric}\n")
                    for i, item in enumerate(ranking[:3], 1):
                        f.write(f"{i}. {item['method']}: {item['value']:.6f}\n")
                    f.write("\n")
        
        logger.info(f"サマリーレポートを保存: {report_file}")


def demonstrate_comprehensive_analysis():
    """包括的分析のデモンストレーション"""
    from src.customer.dimension_reduction.utils.sample_data_generator import CustomerSampleGenerator, CustomerDataConfig
    
    print("=== 次元削減手法包括比較デモ ===")
    
    # より大きなサンプルデータで比較の意味を出す
    config = CustomerDataConfig(
        n_customers=3000, 
        n_products=1000, 
        n_latent_factors=8,
        sparsity_rate=0.9
    )
    generator = CustomerSampleGenerator(config)
    sample_data = generator.save_sample_data("data/customer/samples")
    
    X = sample_data["purchase_matrix"]
    
    print(f"分析データサイズ: {X.shape}")
    print(f"真の潜在因子数: {config.n_latent_factors}")
    print(f"スパース率: {(X == 0).mean():.1%}")
    
    # 包括分析の実行
    analysis_config = AnalysisConfig(
        n_components=12,
        test_size_ratio=0.25,
        save_results=True
    )
    
    comparator = DimensionReductionComparator(analysis_config)
    results = comparator.comprehensive_analysis(X)
    
    # 結果表示
    print("\n=== 分析結果サマリー ===")
    
    if "recommendation" in results:
        primary_rec = results["recommendation"]["primary_recommendation"]
        if primary_rec:
            print(f"推奨手法: {primary_rec['method']}")
            print(f"推奨理由: {primary_rec['reason']}")
            print(f"信頼度: {primary_rec['confidence']:.1%}")
    
    print("\n=== 手法別性能 ===")
    for method, result in results.get("individual_results", {}).items():
        if "error" not in result:
            perf = result.get("performance_metrics", {})
            eff = result.get("efficiency_metrics", {})
            
            print(f"\n{method}:")
            if "test_reconstruction_error" in perf:
                print(f"  復元誤差: {perf['test_reconstruction_error']:.6f}")
            if "total_variance_explained" in perf:
                print(f"  分散説明: {perf['total_variance_explained']:.1%}")
            if "fit_time" in eff:
                print(f"  計算時間: {eff['fit_time']:.2f}秒")
    
    return comparator, results


if __name__ == "__main__":
    demonstrate_comprehensive_analysis()
