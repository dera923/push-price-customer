"""
Customer Sample Data Generator for Dimension Reduction Analysis
============================================================

KARTEの実データ構造を模したサンプルデータを生成します。
この実装は、Google/Meta/NASAレベルの分析を可能にする
高品質なシミュレーションデータを提供します。

理論的背景：
- 顧客行動は潜在的な因子（価格感度、カテゴリ嗜好など）によって支配される
- これらの潜在因子は観測される購買パターンに現れる
- 次元削減はこの逆変換を行い、潜在構造を発見する
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CustomerDataConfig:
    """顧客データ生成の設定クラス"""
    n_customers: int = 50000  # 顧客数（実際の1700万の縮小版）
    n_products: int = 10000   # 商品数
    n_categories: int = 50    # カテゴリ数
    n_latent_factors: int = 8 # 潜在因子数（真の次元）
    sparsity_rate: float = 0.95  # スパース性（95%の要素が0）
    noise_std: float = 0.1    # ノイズの標準偏差
    time_periods: int = 365   # 観測期間（日）
    random_seed: int = 42


class CustomerSampleGenerator:
    """
    顧客行動の真の潜在構造を持つサンプルデータ生成器
    
    この実装の核心的アイデア：
    1. 真の潜在因子を設計（価格感度、ブランド志向など）
    2. 顧客×因子、因子×商品の行列を生成
    3. 行列積で顧客×商品の購買傾向行列を構築
    4. 現実的なノイズとスパース性を追加
    
    これにより、次元削減アルゴリズムが「発見すべき真の構造」を
    事前に知っている状態でテストできます。
    """
    
    def __init__(self, config: CustomerDataConfig):
        self.config = config
        np.random.seed(config.random_seed)
        logger.info(f"CustomerSampleGenerator初期化: {config.n_customers}顧客, {config.n_products}商品")
    
    def generate_latent_factors(self) -> Dict[str, np.ndarray]:
        """
        潜在因子の生成
        
        Returns:
            顧客因子行列(U)と商品因子行列(V)のペア
            U: (n_customers, n_latent_factors)
            V: (n_latent_factors, n_products)
        """
        # 因子の意味を明確に定義
        factor_names = [
            "price_sensitivity",    # 価格感度
            "brand_loyalty",       # ブランド志向
            "category_food",       # 食品カテゴリ嗜好
            "category_beauty",     # 美容カテゴリ嗜好
            "frequency_high",      # 高頻度購買
            "seasonal_winter",     # 冬季購買
            "premium_preference",  # プレミアム志向
            "impulse_buying"       # 衝動買い傾向
        ]
        
        logger.info("潜在因子の生成開始...")
        
        # 顧客因子行列 U の生成（現実的な分布を考慮）
        U = np.zeros((self.config.n_customers, self.config.n_latent_factors))
        
        for i, factor in enumerate(factor_names):
            if "sensitivity" in factor or "loyalty" in factor:
                # 価格感度やブランド志向は二峰性分布（両極端に分かれる）
                U[:, i] = np.concatenate([
                    np.random.normal(-1.5, 0.5, self.config.n_customers // 2),
                    np.random.normal(1.5, 0.5, self.config.n_customers // 2)
                ])
            elif "category" in factor:
                # カテゴリ嗜好は正規分布（一部の人が強く嗜好）
                U[:, i] = np.random.exponential(0.5, self.config.n_customers) * \
                         np.random.choice([-1, 1], self.config.n_customers)
            else:
                # その他は標準正規分布
                U[:, i] = np.random.normal(0, 1, self.config.n_customers)
        
        # 商品因子行列 V の生成
        V = np.random.normal(0, 1, (self.config.n_latent_factors, self.config.n_products))
        
        # カテゴリ構造の注入（商品をカテゴリごとにクラスタ化）
        products_per_category = self.config.n_products // self.config.n_categories
        for cat in range(self.config.n_categories):
            start_idx = cat * products_per_category
            end_idx = min((cat + 1) * products_per_category, self.config.n_products)
            
            # このカテゴリの商品は特定の因子で高い値を持つ
            primary_factor = cat % self.config.n_latent_factors
            V[primary_factor, start_idx:end_idx] *= 2.0  # 因子の影響を強める
        
        logger.info(f"潜在因子生成完了: U{U.shape}, V{V.shape}")
        
        return {
            "customer_factors": U,
            "product_factors": V,
            "factor_names": factor_names
        }
    
    def generate_customer_product_matrix(self, factors: Dict[str, np.ndarray]) -> np.ndarray:
        """
        顧客×商品の購買傾向行列を生成
        
        Args:
            factors: generate_latent_factors()の出力
        
        Returns:
            購買傾向行列 (n_customers, n_products)
        """
        U = factors["customer_factors"]
        V = factors["product_factors"]
        
        logger.info("顧客×商品行列の生成開始...")
        
        # 基本の購買傾向： X = U @ V
        X_base = U @ V
        
        # 現実的な非線形性を追加
        # 購買は確率的で、負の値は意味がない
        X_prob = 1.0 / (1.0 + np.exp(-X_base))  # シグモイド変換
        
        # スパース性の注入（ほとんどの顧客は少数の商品しか買わない）
        mask = np.random.random(X_prob.shape) > self.config.sparsity_rate
        X_sparse = X_prob * mask
        
        # ノイズの追加（測定誤差や未観測要因）
        noise = np.random.normal(0, self.config.noise_std, X_sparse.shape)
        X_final = np.maximum(0, X_sparse + noise)  # 負値を0にクリップ
        
        logger.info(f"購買行列生成完了: {X_final.shape}, スパース率: {(X_final == 0).mean():.3f}")
        
        return X_final
    
    def generate_customer_attributes(self) -> pd.DataFrame:
        """
        majica_member_information テーブルライクな顧客属性を生成
        
        Returns:
            顧客属性のDataFrame
        """
        logger.info("顧客属性データ生成開始...")
        
        # 地域分布（実際のドンキホーテの店舗分布を参考）
        cities = ["Tokyo", "Osaka", "Nagoya", "Yokohama", "Sapporo", "Fukuoka", "Sendai", "Hiroshima"]
        city_weights = [0.25, 0.15, 0.1, 0.1, 0.08, 0.08, 0.06, 0.18]  # 残りは「その他」
        
        # 職業分布
        occupations = ["office_worker", "student", "housewife", "part_time", "self_employed", "retired", "other"]
        occupation_weights = [0.35, 0.15, 0.2, 0.15, 0.08, 0.05, 0.02]
        
        # 会員ランク（購買頻度・金額ベース）
        member_ranks = ["bronze", "silver", "gold", "platinum", "diamond"]
        rank_weights = [0.4, 0.3, 0.2, 0.08, 0.02]
        
        attributes = []
        
        for i in range(self.config.n_customers):
            # hash_idの生成（実際のような16進数文字列）
            hash_id = f"cust_{i:08d}_{np.random.randint(10000, 99999):05d}"
            
            # 地域選択
            city = np.random.choice(cities + ["other"], p=city_weights + [1-sum(city_weights)])
            
            # 職業選択
            occupation = np.random.choice(occupations, p=occupation_weights)
            
            # 会員ランク選択
            member_rank = np.random.choice(member_ranks, p=rank_weights)
            
            # 登録日（過去2年間でランダム）
            register_date = datetime.now() - timedelta(days=np.random.randint(1, 730))
            
            # その他の属性
            attributes.append({
                "hash_id": hash_id,
                "city": city,
                "occupation": occupation,
                "member_rank": member_rank,
                "member_rank_name": f"{member_rank}_member",
                "register_status": "active" if np.random.random() > 0.05 else "inactive",
                "use_store_code": f"store_{np.random.randint(1, 500):03d}",
                "register_date": register_date,
                "bounce_class_cdm": np.random.choice(["A", "B", "C", "D"], p=[0.1, 0.3, 0.4, 0.2])
            })
        
        df_customers = pd.DataFrame(attributes)
        
        logger.info(f"顧客属性生成完了: {len(df_customers)}件")
        
        return df_customers
    
    def generate_product_master(self) -> pd.DataFrame:
        """
        商品マスター情報を生成
        
        Returns:
            商品マスターのDataFrame
        """
        logger.info("商品マスター生成開始...")
        
        # カテゴリ定義
        categories = [
            "food_fresh", "food_frozen", "food_snack", "beverage_soft", "beverage_alcohol",
            "beauty_skincare", "beauty_makeup", "beauty_hair", "fashion_clothes", "fashion_accessories",
            "household_cleaning", "household_kitchen", "health_medicine", "health_supplement",
            "entertainment_books", "entertainment_games", "electronics_mobile", "electronics_pc",
            "pet_food", "pet_accessories", "baby_clothes", "baby_food", "sports_wear", "sports_equipment",
            "travel_goods", "stationery", "gift_seasonal", "gift_celebration", "other_misc"
        ]
        
        products = []
        
        for i in range(self.config.n_products):
            product_id = f"prod_{i:06d}"
            category = categories[i % len(categories)]
            
            # カテゴリごとの価格帯設定
            if "food" in category:
                base_price = np.random.lognormal(np.log(300), 0.5)  # 平均300円程度
            elif "beauty" in category:
                base_price = np.random.lognormal(np.log(1500), 0.7)  # 平均1500円程度
            elif "electronics" in category:
                base_price = np.random.lognormal(np.log(8000), 1.0)  # 平均8000円程度
            else:
                base_price = np.random.lognormal(np.log(800), 0.6)   # 平均800円程度
            
            # コスト（価格の60-80%）
            cost = base_price * np.random.uniform(0.6, 0.8)
            
            products.append({
                "product_id": product_id,
                "category": category,
                "base_price": round(base_price),
                "cost": round(cost),
                "margin_rate": (base_price - cost) / base_price
            })
        
        df_products = pd.DataFrame(products)
        
        logger.info(f"商品マスター生成完了: {len(df_products)}件")
        
        return df_products
    
    def save_sample_data(self, output_dir: str = "data/customer/samples"):
        """
        生成したサンプルデータを保存
        
        Args:
            output_dir: 出力ディレクトリ
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("サンプルデータ生成・保存開始...")
        
        # 1. 潜在因子の生成
        factors = self.generate_latent_factors()
        
        # 2. 購買行列の生成
        purchase_matrix = self.generate_customer_product_matrix(factors)
        
        # 3. 顧客属性の生成
        df_customers = self.generate_customer_attributes()
        
        # 4. 商品マスターの生成
        df_products = self.generate_product_master()
        
        # 5. データ保存
        # 購買行列（NumPy形式とスパース形式の両方）
        np.save(f"{output_dir}/purchase_matrix_dense.npy", purchase_matrix)
        
        # スパース形式での保存（メモリ効率）
        sparse_matrix = sp.csr_matrix(purchase_matrix)
        sp.save_npz(f"{output_dir}/purchase_matrix_sparse.npz", sparse_matrix)
        
        # 真の潜在因子（検証用）
        np.save(f"{output_dir}/true_customer_factors.npy", factors["customer_factors"])
        np.save(f"{output_dir}/true_product_factors.npy", factors["product_factors"])
        
        # 属性データ
        df_customers.to_parquet(f"{output_dir}/customer_attributes.parquet")
        df_products.to_parquet(f"{output_dir}/product_master.parquet")
        
        # メタデータ
        metadata = {
            "generation_time": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "factor_names": factors["factor_names"],
            "data_shape": {
                "customers": len(df_customers),
                "products": len(df_products),
                "purchase_matrix_shape": purchase_matrix.shape,
                "sparsity_rate": float((purchase_matrix == 0).mean()),
                "true_rank": self.config.n_latent_factors
            }
        }
        
        import json
        with open(f"{output_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"サンプルデータ保存完了: {output_dir}")
        logger.info(f"購買行列サイズ: {purchase_matrix.shape}")
        logger.info(f"実際のスパース率: {(purchase_matrix == 0).mean():.3f}")
        logger.info(f"真のランク: {self.config.n_latent_factors}")
        
        return {
            "purchase_matrix": purchase_matrix,
            "customer_attributes": df_customers,
            "product_master": df_products,
            "true_factors": factors,
            "metadata": metadata
        }


def main():
    """メイン実行関数"""
    # 設定
    config = CustomerDataConfig(
        n_customers=10000,    # 開発用に小さめに設定
        n_products=2000,
        n_categories=20,
        n_latent_factors=8,   # 解釈しやすい数に
        sparsity_rate=0.92,
        noise_std=0.05
    )
    
    # データ生成器の作成
    generator = CustomerSampleGenerator(config)
    
    # データ生成と保存
    sample_data = generator.save_sample_data()
    
    print("\n=== サンプルデータ生成完了 ===")
    print(f"顧客数: {config.n_customers:,}")
    print(f"商品数: {config.n_products:,}")
    print(f"真の潜在因子数: {config.n_latent_factors}")
    print(f"生成された購買行列のスパース率: {(sample_data['purchase_matrix'] == 0).mean():.1%}")
    print("\nファイル保存場所: data/customer/samples/")


if __name__ == "__main__":
    main()
