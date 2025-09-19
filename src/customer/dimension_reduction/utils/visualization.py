"""
Advanced Visualization Engine for Dimension Reduction
==================================================

次元削減の理論と実践を視覚的に結ぶ最高レベルの可視化システム

可視化哲学：
1. 数学的美しさ：固有値・特異値の分布を美しく表現
2. 直感的理解：複雑な概念を誰でも理解できる形で
3. 実用的洞察：ビジネス判断に直結する示唆を提供
4. 包括的比較：複数手法を同一画面で比較

Google/Meta/NASAレベルの特徴：
- インタラクティブな探索的分析
- 理論的背景の可視的説明
- 大規模データ対応の効率的描画
- 論文品質の美しいグラフィック
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import warnings

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class DimensionReductionVisualizer:
    """
    次元削減結果の高度可視化クラス
    
    可視化カテゴリ：
    1. 理論的可視化：特異値分布、寄与率、収束曲線
    2. 比較可視化：手法間パフォーマンス比較
    3. 探索的可視化：低次元埋め込み、クラスター構造
    4. ビジネス可視化：解釈可能な因子、セグメント分析
    """
    
    def __init__(self, figsize_default: Tuple[int, int] = (12, 8)):
        self.figsize_default = figsize_default
        self.color_palette = sns.color_palette("husl", 10)
        
        # カスタムカラーマップ
        self.custom_cmap = LinearSegmentedColormap.from_list(
            "custom", ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
        )
        
        logger.info("DimensionReductionVisualizer初期化完了")
    
    def plot_comprehensive_comparison(self, results: Dict[str, Any], 
                                    save_path: Optional[str] = None) -> go.Figure:
        """
        包括的比較結果の統合可視化
        
        Args:
            results: comprehensive_analysisの結果
            save_path: 保存パス
            
        Returns:
            Plotlyの統合図表
        """
        logger.info("包括的比較可視化を開始")
        
        # サブプロット構成（2x3グリッド）
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "復元精度比較", "計算効率比較",
                "分散説明力比較", "メモリ使用量比較", 
                "総合スコア比較", "トレードオフ分析"
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        individual_results = results.get("individual_results", {})
        comparison = results.get("comparison", {})
        
        # 有効な結果のみ抽出
        valid_methods = [k for k, v in individual_results.items() if "error" not in v]
        colors = px.colors.qualitative.Set3[:len(valid_methods)]
        
        # 1. 復元精度比較（行1, 列1）
        accuracy_data = []
        for method in valid_methods:
            perf = individual_results[method].get("performance_metrics", {})
            if "test_reconstruction_error" in perf:
                accuracy_data.append(perf["test_reconstruction_error"])
            else:
                accuracy_data.append(np.nan)
        
        fig.add_trace(
            go.Bar(x=valid_methods, y=accuracy_data, name="復元誤差",
                   marker_color=colors[:len(valid_methods)]),
            row=1, col=1
        )
        
        # 2. 計算効率比較（行1, 列2）
        efficiency_data = []
        for method in valid_methods:
            eff = individual_results[method].get("efficiency_metrics", {})
            if "fit_time" in eff:
                efficiency_data.append(eff["fit_time"])
            else:
                efficiency_data.append(np.nan)
        
        fig.add_trace(
            go.Bar(x=valid_methods, y=efficiency_data, name="学習時間",
                   marker_color=colors[:len(valid_methods)]),
            row=1, col=2
        )
        
        # 3. 分散説明力比較（行2, 列1）
        for i, method in enumerate(valid_methods):
            perf = individual_results[method].get("performance_metrics", {})
            if "cumulative_variance_ratio" in perf:
                cumsum_var = perf["cumulative_variance_ratio"][:10]  # 上位10成分
                fig.add_trace(
                    go.Scatter(x=list(range(1, len(cumsum_var)+1)), 
                              y=cumsum_var, name=f"{method}累積寄与率",
                              line_color=colors[i]),
                    row=2, col=1
                )
        
        # 4. メモリ使用量比較（行2, 列2）
        memory_data = []
        for method in valid_methods:
            eff = individual_results[method].get("efficiency_metrics", {})
            if "memory_usage_mb" in eff:
                memory_data.append(eff["memory_usage_mb"])
            else:
                memory_data.append(np.nan)
        
        fig.add_trace(
            go.Bar(x=valid_methods, y=memory_data, name="メモリ使用量",
                   marker_color=colors[:len(valid_methods)]),
            row=2, col=2
        )
        
        # 5. 総合スコア比較（行3, 列1）
        trade_off_analysis = comparison.get("trade_off_analysis", {})
        if trade_off_analysis:
            overall_scores = [trade_off_analysis[method]["overall_score"] 
                            for method in valid_methods 
                            if method in trade_off_analysis]
            
            fig.add_trace(
                go.Bar(x=valid_methods[:len(overall_scores)], y=overall_scores, 
                       name="総合スコア", marker_color=colors[:len(overall_scores)]),
                row=3, col=1
            )
        
        # 6. トレードオフ分析（行3, 列2）
        if trade_off_analysis:
            accuracy_scores = []
            efficiency_scores = []
            interpretability_scores = []
            
            for method in valid_methods:
                if method in trade_off_analysis:
                    ta = trade_off_analysis[method]
                    accuracy_scores.append(ta["accuracy_score"])
                    efficiency_scores.append(ta["efficiency_score"])
                    interpretability_scores.append(ta["interpretability_score"])
            
            fig.add_trace(
                go.Scatter(x=accuracy_scores, y=efficiency_scores,
                          mode='markers+text', text=valid_methods[:len(accuracy_scores)],
                          name="精度vs効率", marker_size=np.array(interpretability_scores)*20+5,
                          marker_color=colors[:len(accuracy_scores)]),
                row=3, col=2
            )
        
        # レイアウト調整
        fig.update_layout(
            title="次元削減手法包括比較ダッシュボード",
            height=1200,
            showlegend=False,
            template="plotly_white"
        )
        
        # Y軸ラベル
        fig.update_yaxes(title_text="復元誤差", row=1, col=1)
        fig.update_yaxes(title_text="時間(秒)", row=1, col=2)
        fig.update_yaxes(title_text="累積寄与率", row=2, col=1)
        fig.update_yaxes(title_text="メモリ(MB)", row=2, col=2)
        fig.update_yaxes(title_text="総合スコア", row=3, col=1)
        fig.update_yaxes(title_text="効率スコア", row=3, col=2)
        fig.update_xaxes(title_text="精度スコア", row=3, col=2)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"包括比較図を保存: {save_path}")
        
        return fig
    
    def plot_singular_value_analysis(self, models: Dict[str, Any], 
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        特異値・固有値分析の可視化
        
        理論的意味：
        - 特異値の減衰：データの本質的次元
        - エルボー点：適切な成分数の判定
        - ギャップ：ノイズと信号の分離
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("特異値・固有値分析", fontsize=16, fontweight='bold')
        
        # 1. 特異値分布（左上）
        ax1 = axes[0, 0]
        for i, (method, model) in enumerate(models.items()):
            if hasattr(model, 's_') and model.s_ is not None:
                singular_values = model.s_
                ax1.semilogy(range(1, len(singular_values)+1), singular_values, 
                           'o-', label=f"{method}", color=self.color_palette[i],
                           linewidth=2, markersize=6)
        
        ax1.set_xlabel("成分番号")
        ax1.set_ylabel("特異値 (対数スケール)")
        ax1.set_title("特異値の分布")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # エルボー点の自動検出と表示
        for method, model in models.items():
            if hasattr(model, 's_') and model.s_ is not None:
                elbow_point = self._detect_elbow_point(model.s_)
                ax1.axvline(x=elbow_point, color='red', linestyle='--', alpha=0.7)
                ax1.text(elbow_point, max(model.s_)/2, f"エルボー点\n(成分{elbow_point})", 
                        ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                break  # 最初の有効なモデルのみ
        
        # 2. 寄与率分布（右上）
        ax2 = axes[0, 1]
        for i, (method, model) in enumerate(models.items()):
            if hasattr(model, 'explained_variance_ratio_') and model.explained_variance_ratio_ is not None:
                var_ratio = model.explained_variance_ratio_
                cumsum_var = np.cumsum(var_ratio)
                
                ax2.bar(range(1, len(var_ratio)+1), var_ratio, alpha=0.7, 
                       label=f"{method} 個別寄与率", color=self.color_palette[i])
                ax2.plot(range(1, len(cumsum_var)+1), cumsum_var, 
                        'o-', label=f"{method} 累積寄与率", color=self.color_palette[i])
        
        ax2.set_xlabel("主成分番号")
        ax2.set_ylabel("寄与率")
        ax2.set_title("分散寄与率分析")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 条件数分析（左下）
        ax3 = axes[1, 0]
        condition_numbers = []
        method_names = []
        
        for method, model in models.items():
            if hasattr(model, 's_') and model.s_ is not None:
                s = model.s_
                cond_num = s[0] / s[-1] if s[-1] > 0 else np.inf
                condition_numbers.append(cond_num)
                method_names.append(method)
        
        if condition_numbers:
            bars = ax3.bar(method_names, condition_numbers, color=self.color_palette[:len(method_names)])
            ax3.set_ylabel("条件数")
            ax3.set_title("行列の条件数比較")
            ax3.set_yscale('log')
            
            # 条件数の良好性ラインを追加
            ax3.axhline(y=1e12, color='red', linestyle='--', alpha=0.7, label='数値的限界')
            ax3.axhline(y=1e6, color='orange', linestyle='--', alpha=0.7, label='注意レベル')
            ax3.legend()
            
            # 値をバーの上に表示
            for bar, cond_num in zip(bars, condition_numbers):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height*1.1,
                        f'{cond_num:.2e}', ha='center', va='bottom')
        
        # 4. スペクトラムギャップ分析（右下）
        ax4 = axes[1, 1]
        for i, (method, model) in enumerate(models.items()):
            if hasattr(model, 's_') and model.s_ is not None:
                s = model.s_
                if len(s) > 1:
                    gaps = s[:-1] - s[1:]  # 連続する特異値の差
                    ax4.plot(range(1, len(gaps)+1), gaps, 'o-', 
                           label=f"{method} ギャップ", color=self.color_palette[i])
        
        ax4.set_xlabel("成分番号")
        ax4.set_ylabel("スペクトラムギャップ")
        ax4.set_title("スペクトラムギャップ分析")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"特異値分析図を保存: {save_path}")
        
        return fig
    
    def plot_factor_interpretation_nmf(self, nmf_model: Any, 
                                     feature_names: Optional[List[str]] = None,
                                     top_k: int = 10, save_path: Optional[str] = None) -> plt.Figure:
        """
        NMF因子の解釈可能性可視化
        
        Args:
            nmf_model: 学習済みNMFモデル
            feature_names: 特徴量名リスト
            top_k: 上位何個の特徴を表示するか
            save_path: 保存パス
        """
        if not hasattr(nmf_model, 'components_') or nmf_model.components_ is None:
            raise ValueError("NMFモデルに因子行列がありません")
        
        H = nmf_model.components_
        n_components, n_features = H.shape
        
        # 表示する成分数を制限
        display_components = min(n_components, 6)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("NMF因子解釈分析", fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for i in range(display_components):
            ax = axes[i]
            
            # 各因子で重要な特徴量のトップk
            factor_values = H[i, :]
            top_indices = np.argsort(factor_values)[-top_k:][::-1]
            
            if feature_names is not None:
                labels = [feature_names[idx] if idx < len(feature_names) else f"特徴{idx}" 
                         for idx in top_indices]
            else:
                labels = [f"特徴{idx}" for idx in top_indices]
            
            top_values = factor_values[top_indices]
            
            # 横棒グラフ
            bars = ax.barh(range(top_k), top_values, color=self.color_palette[i])
            ax.set_yticks(range(top_k))
            ax.set_yticklabels(labels)
            ax.set_xlabel("因子重み")
            ax.set_title(f"因子 {i+1}\n(重要度: {np.sum(factor_values):.3f})")
            
            # 値をバーの右側に表示
            for j, (bar, value) in enumerate(zip(bars, top_values)):
                ax.text(value + max(top_values)*0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', va='center', ha='left', fontsize=9)
        
        # 余った subplot を非表示
        for i in range(display_components, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"NMF因子解釈図を保存: {save_path}")
        
        return fig
    
    def plot_2d_embedding(self, X_transformed: np.ndarray, 
                         method_name: str, 
                         labels: Optional[np.ndarray] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        2次元埋め込み可視化
        
        Args:
            X_transformed: 変換後データ (n_samples, n_components)
            method_name: 手法名
            labels: サンプルラベル（色分け用）
            save_path: 保存パス
        """
        if X_transformed.shape[1] < 2:
            raise ValueError("2次元可視化には少なくとも2成分が必要です")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"{method_name}による2次元埋め込み可視化", fontsize=14, fontweight='bold')
        
        # 1. 散布図（左）
        ax1 = axes[0]
        if labels is not None:
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax1.scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                           c=self.color_palette[i % len(self.color_palette)], 
                           label=f"クラスター {label}", alpha=0.7, s=30)
            ax1.legend()
        else:
            ax1.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                       alpha=0.7, s=30, c='steelblue')
        
        ax1.set_xlabel("第1主成分")
        ax1.set_ylabel("第2主成分")
        ax1.set_title("主成分散布図")
        ax1.grid(True, alpha=0.3)
        
        # 2. 密度分布（右）
        ax2 = axes[1]
        try:
            # KDEプロット
            from scipy.stats import gaussian_kde
            
            # データの範囲を設定
            x_min, x_max = X_transformed[:, 0].min(), X_transformed[:, 0].max()
            y_min, y_max = X_transformed[:, 1].min(), X_transformed[:, 1].max()
            
            # グリッド作成
            xx, yy = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            # 密度推定
            kernel = gaussian_kde(X_transformed[:, :2].T)
            density = np.reshape(kernel(positions).T, xx.shape)
            
            # 等高線プロット
            contour = ax2.contourf(xx, yy, density, levels=20, cmap=self.custom_cmap, alpha=0.8)
            ax2.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                       alpha=0.6, s=20, c='white', edgecolors='black', linewidth=0.5)
            
            plt.colorbar(contour, ax=ax2, label='密度')
            ax2.set_xlabel("第1主成分")
            ax2.set_ylabel("第2主成分")
            ax2.set_title("密度分布")
            
        except ImportError:
            # scipy.statsが使えない場合はヒストグラム
            ax2.hist2d(X_transformed[:, 0], X_transformed[:, 1], bins=30, cmap=self.custom_cmap)
            ax2.set_xlabel("第1主成分")
            ax2.set_ylabel("第2主成分")
            ax2.set_title("2次元ヒストグラム")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"2次元埋め込み図を保存: {save_path}")
        
        return fig
    
    def plot_performance_radar(self, comparison_results: Dict[str, Any], 
                             save_path: Optional[str] = None) -> go.Figure:
        """
        レーダーチャートによる手法比較
        
        Args:
            comparison_results: 比較分析結果
            save_path: 保存パス
        """
        trade_off_analysis = comparison_results.get("trade_off_analysis", {})
        if not trade_off_analysis:
            raise ValueError("トレードオフ分析結果が必要です")
        
        methods = list(trade_off_analysis.keys())
        metrics = ["accuracy_score", "efficiency_score", "interpretability_score", "overall_score"]
        metric_labels = ["精度", "効率", "解釈性", "総合"]
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3[:len(methods)]
        
        for i, method in enumerate(methods):
            values = [trade_off_analysis[method][metric] for metric in metrics]
            # レーダーチャートを閉じるために最初の値を末尾に追加
            values.append(values[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_labels + [metric_labels[0]],  # 閉じるため
                fill='toself',
                name=method,
                line_color=colors[i],
                fillcolor=colors[i],
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickmode='linear',
                    dtick=0.2
                )
            ),
            showlegend=True,
            title="次元削減手法性能レーダーチャート",
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"レーダーチャートを保存: {save_path}")
        
        return fig
    
    def _detect_elbow_point(self, singular_values: np.ndarray) -> int:
        """
        エルボー点の自動検出
        
        Args:
            singular_values: 特異値配列（降順）
            
        Returns:
            エルボー点のインデックス
        """
        if len(singular_values) < 3:
            return 1
        
        # 二次差分を計算
        diffs = np.diff(singular_values)
        second_diffs = np.diff(diffs)
        
        # 最大の二次差分を持つ点をエルボー点とする
        elbow_idx = np.argmax(second_diffs) + 2  # インデックス調整
        
        return min(elbow_idx, len(singular_values))
    
    def create_interactive_dashboard(self, results: Dict[str, Any], 
                                   models: Dict[str, Any],
                                   X_original: np.ndarray,
                                   save_path: Optional[str] = None) -> str:
        """
        インタラクティブダッシュボードの作成
        
        Args:
            results: 包括分析結果
            models: 学習済みモデル辞書
            X_original: 元データ
            save_path: HTMLファイル保存パス
        
        Returns:
            生成されたHTMLファイルのパス
        """
        if save_path is None:
            save_path = "results/dimension_reduction/interactive_dashboard.html"
        
        # 複数の可視化を組み合わせてダッシュボード作成
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>次元削減分析ダッシュボード</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .chart-container {{ border: 1px solid #ddd; padding: 10px; border-radius: 8px; }}
                h1, h2 {{ color: #333; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>次元削減分析ダッシュボード</h1>
            
            <div class="summary">
                <h2>分析サマリー</h2>
                <p>データサイズ: {X_original.shape[0]:,} × {X_original.shape[1]:,}</p>
                <p>スパース率: {(X_original == 0).mean():.1%}</p>
        """
        
        # 推奨結果の追加
        recommendation = results.get("recommendation", {})
        primary_rec = recommendation.get("primary_recommendation")
        if primary_rec:
            dashboard_html += f"""
                <p><strong>推奨手法:</strong> {primary_rec['method']} (信頼度: {primary_rec['confidence']:.1%})</p>
                <p><strong>理由:</strong> {primary_rec['reason']}</p>
            """
        
        dashboard_html += """
            </div>
            
            <div class="dashboard-container">
                <div class="chart-container">
                    <div id="comparison-chart"></div>
                </div>
                <div class="chart-container">
                    <div id="radar-chart"></div>
                </div>
            </div>
            
            <script>
                // ここに動的なJavaScriptコードを生成
            </script>
        </body>
        </html>
        """
        
        # HTMLファイルの保存
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        logger.info(f"インタラクティブダッシュボードを生成: {save_path}")
        return save_path


def demonstrate_visualization():
    """可視化機能のデモンストレーション"""
    from src.customer.dimension_reduction.core.dimension_reduction_analyzer import demonstrate_comprehensive_analysis
    
    print("=== 高度可視化デモンストレーション ===")
    
    # 包括分析を実行して結果とモデルを取得
    comparator, results = demonstrate_comprehensive_analysis()
    
    # 可視化エンジン作成
    visualizer = DimensionReductionVisualizer()
    
    # 1. 包括比較可視化
    print("包括比較ダッシュボードを生成中...")
    comparison_fig = visualizer.plot_comprehensive_comparison(
        results, 
        save_path="results/dimension_reduction/comprehensive_comparison.html"
    )
    
    # 2. 特異値分析
    print("特異値分析図を生成中...")
    models = comparator.models_
    singular_fig = visualizer.plot_singular_value_analysis(
        models,
        save_path="results/dimension_reduction/singular_value_analysis.png"
    )
    
    # 3. レーダーチャート
    print("性能レーダーチャートを生成中...")
    comparison = results.get("comparison", {})
    if comparison and "trade_off_analysis" in comparison:
        radar_fig = visualizer.plot_performance_radar(
            comparison,
            save_path="results/dimension_reduction/performance_radar.html"
        )
    
    # 4. NMF因子解釈（NMFがある場合）
    if "NMF" in models and hasattr(models["NMF"], 'components_'):
        print("NMF因子解釈図を生成中...")
        nmf_fig = visualizer.plot_factor_interpretation_nmf(
            models["NMF"],
            top_k=8,
            save_path="results/dimension_reduction/nmf_factor_interpretation.png"
        )
    
    print("\n=== 可視化完了 ===")
    print("生成されたファイル:")
    print("- results/dimension_reduction/comprehensive_comparison.html")
    print("- results/dimension_reduction/singular_value_analysis.png")
    print("- results/dimension_reduction/performance_radar.html")
    if "NMF" in models:
        print("- results/dimension_reduction/nmf_factor_interpretation.png")
    
    return visualizer, comparison_fig


if __name__ == "__main__":
    demonstrate_visualization()
