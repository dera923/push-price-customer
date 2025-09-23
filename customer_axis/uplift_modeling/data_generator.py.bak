"""
KARTEテーブル構造に基づくリアルサンプルデータ生成器
Google・Meta・NASA級の処置効果推定のための精密サンプル

実装者：データサイエンティスト向け即戦力コード
参考：Imbens&Rubin "Causal Inference" Ch19, Murphy Ch21
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import uuid
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class KARTEDataGenerator:
    """
    KARTEの実際のテーブル構造に基づくサンプルデータ生成
    
    特徴：
    1. 現実的な顧客行動パターンの再現
    2. 処置効果の異質性（heterogeneous treatment effect）を組み込み
    3. Selection biasとConfoundingを意図的に作成
    4. Google/Meta級の評価指標に対応
    """
    
    def __init__(self, n_customers=10000, n_days=90, seed=42):
        np.random.seed(seed)
        self.n_customers = n_customers
        self.n_days = n_days
        self.start_date = datetime(2024, 7, 1)
        self.end_date = self.start_date + timedelta(days=n_days-1)
        
        # 現実的な顧客セグメント分布（マーケティング実務に基づく）
        self.customer_segments = {
            'high_value': 0.15,    # 高価値顧客：15%
            'regular': 0.60,       # 一般顧客：60%
            'dormant': 0.20,       # 休眠顧客：20%
            'new': 0.05           # 新規顧客：5%
        }
    
    def generate_customer_master(self):
        """
        majica_member_information相当のサンプル生成
        
        重要：Selection biasを意図的に作成
        - 高価値顧客ほどPush配信されやすい
        - しかし元々購買率が高いため見かけのUplift効果は低く見える
        """
        customers = []
        
        for i in range(self.n_customers):
            # 顧客IDとhash_id（KARTEの実構造）
            majica_no = f"MJ{i:08d}"
            hash_id = hashlib.md5(majica_no.encode()).hexdigest()[:16]
            
            # セグメント決定（現実的な分布）
            segment_prob = np.random.random()
            if segment_prob < 0.15:
                segment = 'high_value'
                rfm_score = np.random.normal(85, 10)
                loyalty_tier = np.random.choice(['PLATINUM', 'GOLD'], p=[0.7, 0.3])
                age_band = np.random.choice(['30-39', '40-49', '50-59'], p=[0.3, 0.4, 0.3])
            elif segment_prob < 0.75:
                segment = 'regular'
                rfm_score = np.random.normal(60, 15)
                loyalty_tier = np.random.choice(['SILVER', 'BRONZE'], p=[0.6, 0.4])
                age_band = np.random.choice(['20-29', '30-39', '40-49'], p=[0.3, 0.4, 0.3])
            elif segment_prob < 0.95:
                segment = 'dormant'
                rfm_score = np.random.normal(25, 10)
                loyalty_tier = 'BRONZE'
                age_band = np.random.choice(['20-29', '50-59', '60+'], p=[0.2, 0.4, 0.4])
            else:
                segment = 'new'
                rfm_score = np.random.normal(40, 5)
                loyalty_tier = 'BRONZE'
                age_band = np.random.choice(['20-29', '30-39'], p=[0.6, 0.4])
            
            # クリップして現実的な範囲に
            rfm_score = np.clip(rfm_score, 0, 100)
            
            customers.append({
                'hash_id': hash_id,
                'majica_no': majica_no,
                'segment': segment,
                'member_class': loyalty_tier,
                'rfm_score': round(rfm_score, 1),
                'bounce_class_cdm': np.random.choice(['LOW', 'MEDIUM', 'HIGH'], p=[0.6, 0.3, 0.1]),
                'contact_member_class': np.random.choice(['CONTACTABLE', 'LIMITED', 'NO_CONTACT'], p=[0.7, 0.2, 0.1]),
                'mobile_owned_class': np.random.choice(['OWNED', 'NOT_OWNED'], p=[0.85, 0.15]),
                'thankyou_mail_received': np.random.choice(['YES', 'NO'], p=[0.6, 0.4]),
                'newsletter_received': np.random.choice(['YES', 'NO'], p=[0.4, 0.6]),
                'coupon_availability_class': np.random.choice(['AVAILABLE', 'LIMITED'], p=[0.8, 0.2]),
                'age_band': age_band,
                'integrated_datetime': self.start_date - timedelta(days=np.random.randint(30, 1000))
            })
        
        df_customers = pd.DataFrame(customers)
        print(f"✓ 顧客マスタ生成完了: {len(df_customers):,}件")
        print(f"  セグメント分布: {df_customers['segment'].value_counts().to_dict()}")
        
        return df_customers
    
    def generate_push_logs(self, df_customers):
        """
        masspush_event_log_regional相当のサンプル生成
        
        重要：Treatment assignment mechanism
        - RFMスコアが高いほど配信されやすい（Selection bias）
        - しかし完全に決定論的ではない（Randomness保持）
        """
        push_logs = []
        
        # キャンペーン設定（現実的なパターン）
        campaigns = [
            {'campaign_id': 'SUMMER_SALE_001', 'start_day': 10, 'duration': 7, 'target_prob': 0.3},
            {'campaign_id': 'FLASH_SALE_002', 'start_day': 25, 'duration': 3, 'target_prob': 0.5},
            {'campaign_id': 'WEEKLY_RECOMMEND_003', 'start_day': 35, 'duration': 14, 'target_prob': 0.2},
            {'campaign_id': 'REACTIVATION_004', 'start_day': 55, 'duration': 10, 'target_prob': 0.4}
        ]
        
        for _, customer in df_customers.iterrows():
            for campaign in campaigns:
                campaign_start = self.start_date + timedelta(days=campaign['start_day'])
                
                # Selection bias: RFMスコアに基づく配信確率
                # 高RFM → 高配信確率だが、元々購買率も高い
                rfm_factor = (customer['rfm_score'] / 100) ** 0.5
                base_prob = campaign['target_prob']
                treatment_prob = base_prob * (0.5 + 0.5 * rfm_factor)
                treatment_prob = np.clip(treatment_prob, 0.05, 0.95)
                
                # 処置割当（二値）
                is_treated = np.random.random() < treatment_prob
                
                if is_treated:
                    for day_offset in range(campaign['duration']):
                        send_date = campaign_start + timedelta(days=day_offset)
                        if send_date <= self.end_date:
                            
                            # 送信ログ
                            event_hash = hashlib.md5(f"{customer['hash_id']}{send_date}{campaign['campaign_id']}send".encode()).hexdigest()[:12]
                            push_logs.append({
                                'event_hash': event_hash,
                                'event_name': 'message_send',
                                'api_key': 'karte_api_demo',
                                'user_data_list': customer['hash_id'],
                                'timestamp': send_date + timedelta(hours=np.random.randint(9, 18)),
                                'campaign_id': campaign['campaign_id'],
                                'push_content_id': f"CONTENT_{campaign['campaign_id']}",
                                'plugin_type': 'push_notification',
                                'schedule_id': f"SCH_{campaign['campaign_id']}",
                                'schedule_task_id': f"TASK_{day_offset}",
                                'error_type': None
                            })
                            
                            # クリック確率（セグメント依存）
                            if customer['segment'] == 'high_value':
                                click_prob = 0.15
                            elif customer['segment'] == 'regular':
                                click_prob = 0.08
                            elif customer['segment'] == 'dormant':
                                click_prob = 0.03
                            else:  # new
                                click_prob = 0.12
                            
                            # クリックログ生成
                            if np.random.random() < click_prob:
                                click_hash = hashlib.md5(f"{customer['hash_id']}{send_date}{campaign['campaign_id']}click".encode()).hexdigest()[:12]
                                push_logs.append({
                                    'event_hash': click_hash,
                                    'event_name': 'message_click',
                                    'api_key': 'karte_api_demo',
                                    'user_data_list': customer['hash_id'],
                                    'timestamp': send_date + timedelta(hours=np.random.randint(9, 20), minutes=np.random.randint(0, 59)),
                                    'campaign_id': campaign['campaign_id'],
                                    'push_content_id': f"CONTENT_{campaign['campaign_id']}",
                                    'plugin_type': 'push_notification',
                                    'schedule_id': f"SCH_{campaign['campaign_id']}",
                                    'schedule_task_id': f"TASK_{day_offset}",
                                    'error_type': None
                                })
                
                # CONTROL_GROUP（未配信）記録
                else:
                    control_hash = hashlib.md5(f"{customer['hash_id']}{campaign_start}{campaign['campaign_id']}control".encode()).hexdigest()[:12]
                    push_logs.append({
                        'event_hash': control_hash,
                        'event_name': 'control_group',
                        'api_key': 'karte_api_demo',
                        'user_data_list': customer['hash_id'],
                        'timestamp': campaign_start,
                        'campaign_id': campaign['campaign_id'],
                        'push_content_id': 'CONTROL_GROUP',
                        'plugin_type': 'push_notification',
                        'schedule_id': f"SCH_{campaign['campaign_id']}",
                        'schedule_task_id': 'CONTROL',
                        'error_type': None
                    })
        
        df_push_logs = pd.DataFrame(push_logs)
        print(f"✓ Push配信ログ生成完了: {len(df_push_logs):,}件")
        print(f"  イベント分布: {df_push_logs['event_name'].value_counts().to_dict()}")
        
        return df_push_logs
    
    def generate_purchase_data(self, df_customers, df_push_logs):
        """
        pos_trade_item_sales相当のサンプル生成
        
        重要：Heterogeneous Treatment Effect
        - セグメント別に異なるUplift効果
        - Push配信のタイミング効果（短期・中期）
        """
        purchases = []
        
        # 商品カテゴリ（現実的な分布）
        product_categories = {
            'FOOD': {'weight': 0.4, 'avg_price': 500, 'repeat_rate': 0.7},
            'COSMETICS': {'weight': 0.25, 'avg_price': 2000, 'repeat_rate': 0.3},
            'CLOTHING': {'weight': 0.2, 'avg_price': 3500, 'repeat_rate': 0.2},
            'ELECTRONICS': {'weight': 0.1, 'avg_price': 8000, 'repeat_rate': 0.1},
            'OTHER': {'weight': 0.05, 'avg_price': 1200, 'repeat_rate': 0.4}
        }
        
        # Push配信履歴を顧客別に整理
        push_by_customer = df_push_logs.groupby('user_data_list').agg({
            'timestamp': list,
            'event_name': list,
            'campaign_id': list
        }).to_dict('index')
        
        for _, customer in df_customers.iterrows():
            hash_id = customer['hash_id']
            segment = customer['segment']
            rfm_score = customer['rfm_score']
            
            # ベース購買確率（セグメント依存）
            if segment == 'high_value':
                base_purchase_prob = 0.12  # 1日あたり12%
                avg_basket_size = 2.5
            elif segment == 'regular':
                base_purchase_prob = 0.05  # 1日あたり5%
                avg_basket_size = 1.8
            elif segment == 'dormant':
                base_purchase_prob = 0.01  # 1日あたり1%
                avg_basket_size = 1.2
            else:  # new
                base_purchase_prob = 0.03  # 1日あたり3%
                avg_basket_size = 1.5
            
            # 日別の購買シミュレーション
            for day_offset in range(self.n_days):
                current_date = self.start_date + timedelta(days=day_offset)
                
                # その日のPush配信状況チェック
                push_effect = 0.0
                if hash_id in push_by_customer:
                    customer_pushes = push_by_customer[hash_id]
                    for i, push_timestamp in enumerate(customer_pushes['timestamp']):
                        if isinstance(push_timestamp, list):
                            push_timestamps = push_timestamp
                        else:
                            push_timestamps = [push_timestamp]
                        
                        for push_ts in push_timestamps:
                            if isinstance(push_ts, str):
                                push_ts = pd.to_datetime(push_ts)
                            
                            days_since_push = (current_date - push_ts.date()).days
                            
                            # Push効果（時間減衰）
                            if 0 <= days_since_push <= 7:
                                # セグメント別のUplift効果（異質性）
                                if segment == 'high_value':
                                    uplift_base = 0.03  # 3%の追加購買確率
                                elif segment == 'regular':
                                    uplift_base = 0.08  # 8%の追加購買確率（最も効果的）
                                elif segment == 'dormant':
                                    uplift_base = 0.15  # 15%の追加購買確率（休眠覚醒）
                                else:  # new
                                    uplift_base = 0.10  # 10%の追加購買確率
                                
                                # 時間減衰
                                time_decay = np.exp(-days_since_push * 0.3)
                                push_effect += uplift_base * time_decay
                
                # 最終的な購買確率
                final_purchase_prob = base_purchase_prob + push_effect
                final_purchase_prob = np.clip(final_purchase_prob, 0, 0.8)
                
                # 購買判定
                if np.random.random() < final_purchase_prob:
                    # バスケットサイズ決定
                    n_items = max(1, int(np.random.poisson(avg_basket_size)))
                    
                    for item_idx in range(n_items):
                        # 商品カテゴリ選択
                        category = np.random.choice(
                            list(product_categories.keys()),
                            p=[cat['weight'] for cat in product_categories.values()]
                        )
                        cat_info = product_categories[category]
                        
                        # 商品詳細
                        sku_id = f"SKU_{category}_{np.random.randint(1000, 9999)}"
                        
                        # 価格（正規分布 + ログ変換で現実的な分布）
                        price_base = cat_info['avg_price']
                        price_variation = np.random.lognormal(0, 0.5)
                        sales_price = max(100, int(price_base * price_variation))
                        
                        # 割引設定（Push効果で割引率変動）
                        discount_rate = 0.0
                        if push_effect > 0.05:  # 強いPush効果がある場合
                            discount_rate = np.random.uniform(0.05, 0.2)
                        elif push_effect > 0.02:
                            discount_rate = np.random.uniform(0.0, 0.1)
                        
                        final_price = int(sales_price * (1 - discount_rate))
                        
                        purchases.append({
                            'majica_no': customer['majica_no'],
                            'hash_id': hash_id,
                            'date': current_date.date(),
                            'sku_id': sku_id,
                            'category': category,
                            'quantity': 1,
                            'list_price': sales_price,
                            'sales_price': final_price,
                            'discount_amount': sales_price - final_price,
                            'discount_rate': discount_rate,
                            'segment': segment,
                            'rfm_score': rfm_score,
                            'push_effect_score': round(push_effect, 4),
                            'taxfree_group_class': np.random.choice(['TAXFREE', 'TAXABLE'], p=[0.1, 0.9]),
                            'floor_no': np.random.choice(['B1', '1F', '2F', '3F'], p=[0.3, 0.4, 0.2, 0.1]),
                            'customer_quantity_daily': n_items
                        })
        
        df_purchases = pd.DataFrame(purchases)
        print(f"✓ 購買データ生成完了: {len(df_purchases):,}件")
        print(f"  日平均購買件数: {len(df_purchases)/self.n_days:.1f}件")
        print(f"  カテゴリ分布: {df_purchases['category'].value_counts().to_dict()}")
        
        return df_purchases
    
    def create_analysis_dataset(self, df_customers, df_push_logs, df_purchases):
        """
        Upliftモデリング用の統合データセット作成
        
        重要：この統合プロセスがGoogle・Meta・NASAで最も重要
        - 顧客×日付レベルでの観測単位統一
        - Treatment assignment flagの正確な作成
        - Confounding variablesの適切な設計
        """
        analysis_data = []
        
        # Push配信データを日別・顧客別に集約
        push_summary = df_push_logs.groupby(['user_data_list', pd.to_datetime(df_push_logs['timestamp']).dt.date]).agg({
            'event_name': lambda x: 'message_send' in list(x),
            'campaign_id': 'first'
        }).reset_index()
        push_summary.columns = ['hash_id', 'date', 'push_sent', 'campaign_id']
        
        # 購買データを日別・顧客別に集約
        purchase_summary = df_purchases.groupby(['hash_id', 'date']).agg({
            'quantity': 'sum',
            'sales_price': 'sum',
            'discount_amount': 'sum',
            'push_effect_score': 'mean',
            'category': lambda x: ','.join(list(set(x)))
        }).reset_index()
        
        # 顧客×日付の全組み合わせ作成
        all_customers = df_customers['hash_id'].unique()
        all_dates = pd.date_range(self.start_date.date(), self.end_date.date(), freq='D')
        
        for customer_id in all_customers:
            customer_info = df_customers[df_customers['hash_id'] == customer_id].iloc[0]
            
            for date in all_dates:
                # Push処置フラグ
                push_info = push_summary[(push_summary['hash_id'] == customer_id) & 
                                       (push_summary['date'] == date)]
                treatment = 1 if len(push_info) > 0 and push_info.iloc[0]['push_sent'] else 0
                
                # 購買アウトカム
                purchase_info = purchase_summary[(purchase_summary['hash_id'] == customer_id) & 
                                               (purchase_summary['date'] == date)]
                
                if len(purchase_info) > 0:
                    purchase_flag = 1
                    total_spend = purchase_info.iloc[0]['sales_price']
                    total_quantity = purchase_info.iloc[0]['quantity']
                    push_effect = purchase_info.iloc[0]['push_effect_score']
                else:
                    purchase_flag = 0
                    total_spend = 0
                    total_quantity = 0
                    push_effect = 0
                
                # 時間的特徴量
                day_of_week = date.weekday()
                is_weekend = 1 if day_of_week >= 5 else 0
                days_since_start = (date - self.start_date.date()).days
                
                # 過去の購買履歴（ラグ特徴量）
                past_purchases = df_purchases[
                    (df_purchases['hash_id'] == customer_id) & 
                    (df_purchases['date'] < date) &
                    (df_purchases['date'] >= date - timedelta(days=30))
                ]
                past_30d_purchases = len(past_purchases)
                past_30d_spend = past_purchases['sales_price'].sum() if len(past_purchases) > 0 else 0
                
                analysis_data.append({
                    'hash_id': customer_id,
                    'date': date,
                    'treatment': treatment,  # Push配信フラグ（二値処置）
                    'outcome_purchase': purchase_flag,  # 購買フラグ（主要アウトカム）
                    'outcome_spend': total_spend,  # 購買金額
                    'outcome_quantity': total_quantity,  # 購買点数
                    'true_push_effect': push_effect,  # 真のPush効果（評価用）
                    
                    # 顧客属性（Confounders）
                    'segment': customer_info['segment'],
                    'rfm_score': customer_info['rfm_score'],
                    'loyalty_tier': customer_info['member_class'],
                    'age_band': customer_info['age_band'],
                    'mobile_owned': 1 if customer_info['mobile_owned_class'] == 'OWNED' else 0,
                    
                    # 時間特徴量
                    'day_of_week': day_of_week,
                    'is_weekend': is_weekend,
                    'days_since_start': days_since_start,
                    
                    # ラグ特徴量（Selection biasを作る重要な変数）
                    'past_30d_purchases': past_30d_purchases,
                    'past_30d_spend': past_30d_spend,
                    'has_recent_purchase': 1 if past_30d_purchases > 0 else 0
                })
        
        df_analysis = pd.DataFrame(analysis_data)
        
        # カテゴリ変数のエンコーディング
        df_analysis = pd.get_dummies(df_analysis, columns=['segment', 'loyalty_tier', 'age_band'], 
                                   prefix=['seg', 'tier', 'age'], drop_first=True)
        
        print(f"✓ 分析用データセット作成完了: {len(df_analysis):,}件")
        print(f"  処置群比率: {df_analysis['treatment'].mean():.3f}")
        print(f"  購買率（処置群）: {df_analysis[df_analysis['treatment']==1]['outcome_purchase'].mean():.3f}")
        print(f"  購買率（対照群）: {df_analysis[df_analysis['treatment']==0]['outcome_purchase'].mean():.3f}")
        print(f"  素朴な効果差: {df_analysis[df_analysis['treatment']==1]['outcome_purchase'].mean() - df_analysis[df_analysis['treatment']==0]['outcome_purchase'].mean():.4f}")
        
        return df_analysis

def main():
    """サンプルデータ生成のメイン実行"""
    print("🚀 KARTEリアルサンプルデータ生成開始")
    print("=" * 60)
    
    # ディレクトリ作成
    Path("customer_axis/2_2_uplift_modeling/sample_data").mkdir(parents=True, exist_ok=True)
    
    # データ生成器初期化
    generator = KARTEDataGenerator(n_customers=5000, n_days=60, seed=42)
    
    # 各テーブル生成
    print("\n📊 1. 顧客マスタ生成")
    df_customers = generator.generate_customer_master()
    
    print("\n📱 2. Push配信ログ生成")
    df_push_logs = generator.generate_push_logs(df_customers)
    
    print("\n💰 3. 購買データ生成")
    df_purchases = generator.generate_purchase_data(df_customers, df_push_logs)
    
    print("\n🔗 4. 分析用統合データセット作成")
    df_analysis = generator.create_analysis_dataset(df_customers, df_push_logs, df_purchases)
    
    # ファイル保存
    output_dir = Path("customer_axis/2_2_uplift_modeling/sample_data")
    
    df_customers.to_csv(output_dir / "karte_customers.csv", index=False)
    df_push_logs.to_csv(output_dir / "karte_push_logs.csv", index=False)
    df_purchases.to_csv(output_dir / "karte_purchases.csv", index=False)
    df_analysis.to_csv(output_dir / "uplift_analysis_dataset.csv", index=False)
    
    print(f"\n✅ データ生成完了！ファイル保存先: {output_dir}")
    print("\n📋 生成ファイル:")
    print(f"  - karte_customers.csv ({len(df_customers):,} records)")
    print(f"  - karte_push_logs.csv ({len(df_push_logs):,} records)")  
    print(f"  - karte_purchases.csv ({len(df_purchases):,} records)")
    print(f"  - uplift_analysis_dataset.csv ({len(df_analysis):,} records)")
    
    return df_analysis

if __name__ == "__main__":
    df_uplift = main()
