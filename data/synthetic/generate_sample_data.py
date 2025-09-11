"""
KARTE構造を完璧に模擬したサンプルデータ生成器
Google/Meta/NASAレベルのリアルなデータ構造

参照テーブル：
- pos_trade_item_sales (取引データ)
- masspush_event_log_regional (Push配信ログ)
- majica_member_information (顧客情報)
- coupon_actual_history (クーポンデータ)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class KARTEDataSimulator:
    """KARTE実データ構造を模擬するシミュレータ"""
    
    def __init__(self, 
                 n_customers: int = 50000,
                 n_items: int = 1000,
                 n_stores: int = 50,
                 start_date: str = "2024-01-01",
                 end_date: str = "2024-03-31",
                 random_seed: int = 42):
        
        self.n_customers = n_customers
        self.n_items = n_items  
        self.n_stores = n_stores
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        np.random.seed(random_seed)
        
        # 顧客セグメント分布（実際のKARTEに近い分布）
        self.customer_segments = {
            'premium': 0.05,    # 高価値顧客
            'regular': 0.25,    # 定期購入顧客
            'occasional': 0.45, # 時々購入顧客
            'inactive': 0.25    # 非活発顧客
        }
        
    def generate_customer_master(self) -> pd.DataFrame:
        """majica_member_information相当のデータ生成"""
        
        customers = []
        
        for i in range(self.n_customers):
            # セグメント決定
            segment = np.random.choice(
                list(self.customer_segments.keys()),
                p=list(self.customer_segments.values())
            )
            
            # セグメント別の特性設定
            if segment == 'premium':
                age = np.random.normal(45, 10)
                income_level = np.random.choice(['high', 'very_high'], p=[0.3, 0.7])
                app_usage_days = np.random.poisson(25)
            elif segment == 'regular':
                age = np.random.normal(40, 12)
                income_level = np.random.choice(['medium', 'high'], p=[0.6, 0.4])
                app_usage_days = np.random.poisson(15)
            elif segment == 'occasional':
                age = np.random.normal(35, 15)
                income_level = np.random.choice(['low', 'medium'], p=[0.4, 0.6])
                app_usage_days = np.random.poisson(8)
            else:  # inactive
                age = np.random.normal(30, 20)
                income_level = np.random.choice(['low', 'medium'], p=[0.7, 0.3])
                app_usage_days = np.random.poisson(3)
                
            age = max(18, min(80, age))  # 年齢範囲制限
            
            customer = {
                'hash_id': f'cust_{i:06d}',  # 実際のKARTEではハッシュ化ID
                'majica_no': f'M{i:08d}',    # アプリ内顧客番号
                'age': int(age),
                'gender': np.random.choice(['M', 'F'], p=[0.45, 0.55]),
                'prefecture': np.random.choice(['東京都', '大阪府', '神奈川県', '愛知県', 'その他'], 
                                             p=[0.3, 0.15, 0.12, 0.08, 0.35]),
                'income_level': income_level,
                'app_usage_days': max(0, app_usage_days),
                'segment': segment,
                'registration_date': self.start_date - timedelta(days=np.random.randint(30, 1000))
            }
            customers.append(customer)
            
        return pd.DataFrame(customers)
    
    def generate_push_log(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """masspush_event_log_regional相当のデータ生成"""
        
        push_logs = []
        
        # Push配信キャンペーン設定
        campaigns = [
            {'campaign_id': 'PUSH_001', 'name': '春の新商品キャンペーン', 'start': '2024-02-01'},
            {'campaign_id': 'PUSH_002', 'name': 'バレンタインセール', 'start': '2024-02-10'},
            {'campaign_id': 'PUSH_003', 'name': '卒業応援キャンペーン', 'start': '2024-03-01'},
        ]
        
        for campaign in campaigns:
            campaign_start = pd.to_datetime(campaign['start'])
            
            # 配信対象顧客の選択（セグメント別配信確率）
            for _, customer in customers_df.iterrows():
                
                # セグメント別配信確率（実際のマーケティング戦略を反映）
                segment_push_prob = {
                    'premium': 0.9,
                    'regular': 0.8, 
                    'occasional': 0.6,
                    'inactive': 0.3
                }
                
                push_prob = segment_push_prob[customer['segment']]
                
                # 年齢による調整（若い層により多く配信）
                if customer['age'] < 30:
                    push_prob *= 1.2
                elif customer['age'] > 50:
                    push_prob *= 0.8
                    
                push_prob = min(1.0, push_prob)
                
                # 配信判定
                is_pushed = np.random.random() < push_prob
                
                if is_pushed:
                    # 配信時刻生成
                    push_time = campaign_start + timedelta(
                        hours=np.random.randint(0, 72),  # 3日間の配信期間
                        minutes=np.random.randint(0, 60)
                    )
                    
                    # Push送信ログ
                    push_log = {
                        'event_hash': f'evt_{len(push_logs):08d}',
                        'event_name': 'message_send',
                        'timestamp': push_time,
                        'hash_id': customer['hash_id'],  # 顧客ID
                        'campaign_id': campaign['campaign_id'],
                        'push_content_id': f'CONTENT_{campaign["campaign_id"]}',
                        'api_key': 'karte_api_key_masked'
                    }
                    push_logs.append(push_log)
                    
                    # クリック判定（セグメント・年齢による差異）
                    click_prob_base = {
                        'premium': 0.15,
                        'regular': 0.12,
                        'occasional': 0.08,
                        'inactive': 0.04
                    }[customer['segment']]
                    
                    # 年齢調整
                    if customer['age'] < 30:
                        click_prob = click_prob_base * 1.3
                    elif customer['age'] > 50:
                        click_prob = click_prob_base * 0.7
                    else:
                        click_prob = click_prob_base
                        
                    if np.random.random() < click_prob:
                        # Pushクリックログ
                        click_log = {
                            'event_hash': f'evt_{len(push_logs):08d}',
                            'event_name': 'message_click',
                            'timestamp': push_time + timedelta(minutes=np.random.randint(1, 120)),
                            'hash_id': customer['hash_id'],
                            'campaign_id': campaign['campaign_id'],
                            'push_content_id': f'CONTENT_{campaign["campaign_id"]}',
                            'api_key': 'karte_api_key_masked'
                        }
                        push_logs.append(click_log)
                else:
                    # コントロールグループ（未配信層）のマーカー
                    control_log = {
                        'event_hash': f'evt_{len(push_logs):08d}',
                        'event_name': 'message_send',
                        'timestamp': campaign_start,
                        'hash_id': customer['hash_id'],
                        'campaign_id': campaign['campaign_id'],
                        'push_content_id': 'CONTROL_GROUP',  # 未配信のマーカー
                        'api_key': 'karte_api_key_masked'
                    }
                    push_logs.append(control_log)
                    
        return pd.DataFrame(push_logs)
    
    def generate_transaction_data(self, 
                                customers_df: pd.DataFrame,
                                push_logs_df: pd.DataFrame) -> pd.DataFrame:
        """pos_trade_item_sales相当のデータ生成（Push効果を含む）"""
        
        transactions = []
        
        # 商品カテゴリと価格帯の設定
        product_categories = {
            'daily_goods': {'base_price': 300, 'margin': 0.3, 'elasticity': -1.2},
            'fashion': {'base_price': 2000, 'margin': 0.5, 'elasticity': -1.8},
            'electronics': {'base_price': 15000, 'margin': 0.2, 'elasticity': -0.8},
            'food': {'base_price': 800, 'margin': 0.4, 'elasticity': -1.0}
        }
        
        # Push効果の作成（顧客セグメント別）
        push_effect_by_segment = {
            'premium': {'purchase_lift': 1.3, 'amount_lift': 1.2},
            'regular': {'purchase_lift': 1.5, 'amount_lift': 1.3},
            'occasional': {'purchase_lift': 1.8, 'amount_lift': 1.4},
            'inactive': {'purchase_lift': 2.2, 'amount_lift': 1.6}
        }
        
        # 顧客別のPush配信状況を取得
        customer_push_status = {}
        for _, row in push_logs_df.iterrows():
            customer_id = row['hash_id']
            if row['push_content_id'] != 'CONTROL_GROUP':
                customer_push_status[customer_id] = True
            elif customer_id not in customer_push_status:
                customer_push_status[customer_id] = False
                
        # 取引データ生成
        transaction_id = 0
        
        for _, customer in customers_df.iterrows():
            customer_id = customer['hash_id']
            segment = customer['segment']
            
            # Push配信有無
            received_push = customer_push_status.get(customer_id, False)
            
            # セグメント別基本購買頻度
            base_purchase_freq = {
                'premium': 8,
                'regular': 5,
                'occasional': 2,
                'inactive': 0.5
            }[segment]
            
            # Push効果を適用
            if received_push:
                purchase_freq = base_purchase_freq * push_effect_by_segment[segment]['purchase_lift']
                amount_multiplier = push_effect_by_segment[segment]['amount_lift']
            else:
                purchase_freq = base_purchase_freq
                amount_multiplier = 1.0
                
            # 期間内の購買回数決定
            n_purchases = np.random.poisson(purchase_freq)
            
            for purchase_idx in range(n_purchases):
                # 購買日時の生成
                purchase_date = self.start_date + timedelta(
                    days=np.random.randint(0, (self.end_date - self.start_date).days)
                )
                
                # 1回の購買での商品数（1-5点）
                n_items = np.random.randint(1, 6)
                
                for item_idx in range(n_items):
                    transaction_id += 1
                    
                    # 商品カテゴリと価格決定
                    category = np.random.choice(list(product_categories.keys()))
                    product_info = product_categories[category]
                    
                    # 基本価格
                    base_price = product_info['base_price'] * np.random.uniform(0.7, 1.5)
                    
                    # Push効果による購買金額増加
                    actual_price = base_price * amount_multiplier * np.random.uniform(0.9, 1.1)
                    
                    transaction = {
                        'trade_id': f'T{transaction_id:08d}',
                        'hash_id': customer_id,
                        'item_code': f'ITEM_{category}_{np.random.randint(1, 101):03d}',
                        'trade_date': purchase_date.strftime('%Y-%m-%d'),
                        'trade_time': purchase_date.strftime('%H:%M:%S'),
                        'quantity': np.random.randint(1, 4),
                        'list_price': int(base_price),
                        'selling_price': int(actual_price),
                        'discount_amount': max(0, int(base_price - actual_price)),
                        'store_id': f'ST{np.random.randint(1, self.n_stores+1):03d}',
                        'category': category,
                        'received_push': received_push  # 分析用フラグ
                    }
                    
                    transactions.append(transaction)
                    
        return pd.DataFrame(transactions)
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """全データの生成と統合"""
        
        print("顧客マスタデータ生成中...")
        customers_df = self.generate_customer_master()
        
        print("Push配信ログ生成中...")  
        push_logs_df = self.generate_push_log(customers_df)
        
        print("取引データ生成中...")
        transactions_df = self.generate_transaction_data(customers_df, push_logs_df)
        
        # 統合データの作成（TDF：Tabular Data Format）
        print("統合TDF作成中...")
        
        # Push効果分析用の統合データ
        # 各顧客の期間内集計値を計算
        customer_summary = transactions_df.groupby('hash_id').agg({
            'selling_price': ['sum', 'count', 'mean'],
            'quantity': 'sum',
            'received_push': 'first'  # Push受信フラグ
        }).round(2)
        
        # カラム名を平坦化
        customer_summary.columns = ['total_amount', 'purchase_count', 'avg_amount', 'total_quantity', 'treatment']
        customer_summary = customer_summary.reset_index()
        
        # 顧客属性をマージ
        tdf = customer_summary.merge(
            customers_df[['hash_id', 'age', 'gender', 'prefecture', 'income_level', 'segment', 'app_usage_days']],
            on='hash_id',
            how='left'
        )
        
        # カテゴリ変数のエンコーディング
        tdf['gender_M'] = (tdf['gender'] == 'M').astype(int)
        tdf['income_high'] = (tdf['income_level'] == 'high').astype(int) 
        tdf['income_very_high'] = (tdf['income_level'] == 'very_high').astype(int)
        
        # セグメントダミー変数
        for segment in ['premium', 'regular', 'occasional']:
            tdf[f'segment_{segment}'] = (tdf['segment'] == segment).astype(int)
            
        print(f"データ生成完了: 顧客数={len(customers_df)}, 取引数={len(transactions_df)}")
        
        return {
            'customers': customers_df,
            'push_logs': push_logs_df,  
            'transactions': transactions_df,
            'tdf': tdf  # DR-ATE分析用統合データ
        }

# データ生成実行
if __name__ == "__main__":
    simulator = KARTEDataSimulator(n_customers=10000)  # 検証用に小さめ
    data = simulator.generate_all_data()
    
    # データ保存
    data['customers'].to_parquet('data/synthetic/customers.parquet')
    data['push_logs'].to_parquet('data/synthetic/push_logs.parquet') 
    data['transactions'].to_parquet('data/synthetic/transactions.parquet')
    data['tdf'].to_parquet('data/synthetic/tdf_v1.parquet')  # DR-ATE分析用
    
    print("\n=== データ概要 ===")
    print(f"TDF shape: {data['tdf'].shape}")
    print(f"Treatment rate: {data['tdf']['treatment'].mean():.3f}")
    print(f"平均購買額（処置群）: {data['tdf'][data['tdf']['treatment']==1]['total_amount'].mean():.0f}円")
    print(f"平均購買額（対照群）: {data['tdf'][data['tdf']['treatment']==0]['total_amount'].mean():.0f}円")
