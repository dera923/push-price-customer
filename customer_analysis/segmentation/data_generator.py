"""
Customer Sample Data Generator for Segmentation Analysis
=====================================

This module generates realistic customer data that mimics KARTE's actual data structure
Based on the provided schema: majica_member_information, pos_trade_item_sales, etc.

Author: Data Science Team
Purpose: Google/Meta/NASA level customer segmentation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CustomerDataGenerator:
    """
    Generates statistically realistic customer data for segmentation analysis
    
    Based on retail industry benchmarks and KARTE data schema
    """
    
    def __init__(self, n_customers=10000, seed=42):
        """
        Initialize data generator
        
        Args:
            n_customers (int): Number of customers to generate
            seed (int): Random seed for reproducibility
        """
        self.n_customers = n_customers
        self.seed = seed
        np.random.seed(seed)
        
    def generate_customer_master(self):
        """
        Generate customer master data (majica_member_information equivalent)
        
        Returns:
            pd.DataFrame: Customer master data with realistic distributions
        """
        print("🔄 Generating customer master data...")
        
        # Generate customer IDs (hash_id equivalent)
        customer_ids = [f"cust_{str(i).zfill(8)}" for i in range(1, self.n_customers + 1)]
        
        # Age distribution (realistic Japanese demographics)
        ages = np.random.choice(
            range(18, 80), 
            size=self.n_customers,
            p=self._get_age_distribution()
        )
        
        # Gender distribution (52% female, 48% male - retail typical)
        genders = np.random.choice(['F', 'M'], size=self.n_customers, p=[0.52, 0.48])
        
        # Member class (based on RFM-like segmentation)
        member_classes = np.random.choice(
            ['Premium', 'Gold', 'Silver', 'Bronze', 'Basic'],
            size=self.n_customers,
            p=[0.05, 0.15, 0.25, 0.35, 0.20]
        )
        
        # Contact preferences and behavioral flags
        contact_member_class = np.random.choice(['OK', 'NG'], size=self.n_customers, p=[0.7, 0.3])
        mobile_owned_class = np.random.choice(['Y', 'N'], size=self.n_customers, p=[0.85, 0.15])
        newsletter_received = np.random.choice(['Y', 'N'], size=self.n_customers, p=[0.6, 0.4])
        
        # Registration date (spread over 3 years)
        start_date = datetime.now() - timedelta(days=1095)
        registration_dates = [
            start_date + timedelta(days=np.random.randint(0, 1095))
            for _ in range(self.n_customers)
        ]
        
        customer_master = pd.DataFrame({
            'hash_id': customer_ids,
            'age': ages,
            'gender': genders,
            'member_class': member_classes,
            'contact_member_class': contact_member_class,
            'mobile_owned_class': mobile_owned_class,
            'newsletter_received': newsletter_received,
            'integrated_datetime': registration_dates
        })
        
        print(f"✅ Generated {len(customer_master)} customer records")
        return customer_master
    
    def generate_purchase_data(self, customer_master):
        """
        Generate purchase transaction data (pos_trade_item_sales equivalent)
        
        Args:
            customer_master (pd.DataFrame): Customer master data
            
        Returns:
            pd.DataFrame: Purchase transaction data
        """
        print("🔄 Generating purchase transaction data...")
        
        transactions = []
        
        for idx, customer in customer_master.iterrows():
            # Purchase frequency based on member class (Premium = more frequent)
            freq_map = {
                'Premium': (15, 25),  # 15-25 transactions per year
                'Gold': (8, 15),
                'Silver': (4, 8),
                'Bronze': (2, 4),
                'Basic': (1, 3)
            }
            
            min_freq, max_freq = freq_map[customer['member_class']]
            n_transactions = np.random.randint(min_freq, max_freq + 1)
            
            # Generate transactions for this customer
            for _ in range(n_transactions):
                # Transaction date (within last year)
                transaction_date = datetime.now() - timedelta(
                    days=np.random.randint(1, 365)
                )
                
                # Purchase amount (log-normal distribution - typical for retail)
                if customer['member_class'] == 'Premium':
                    amount_params = (7.5, 0.8)  # Higher spending
                elif customer['member_class'] == 'Gold':
                    amount_params = (6.8, 0.9)
                elif customer['member_class'] == 'Silver':
                    amount_params = (6.2, 1.0)
                elif customer['member_class'] == 'Bronze':
                    amount_params = (5.8, 1.1)
                else:  # Basic
                    amount_params = (5.5, 1.2)
                
                sales_amount = np.random.lognormal(*amount_params)
                
                # Quantity (1-5 items typically)
                quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05])
                
                # Calculate other fields
                discount_amount = sales_amount * np.random.uniform(0, 0.2)  # 0-20% discount
                after_compensation_sales_amount = sales_amount - discount_amount
                cost = sales_amount * np.random.uniform(0.6, 0.8)  # 60-80% cost ratio
                grossprofit_amount = sales_amount - cost
                
                transactions.append({
                    'hash_id': customer['hash_id'],
                    'transaction_date': transaction_date,
                    'quantity': quantity,
                    'sales_amount': round(sales_amount, 2),
                    'discount_amount': round(discount_amount, 2),
                    'after_compensation_sales_amount': round(after_compensation_sales_amount, 2),
                    'cost': round(cost, 2),
                    'grossprofit_amount': round(grossprofit_amount, 2),
                    'item_name': f'Item_{np.random.randint(1, 1000)}',
                    'unit_point': round(sales_amount * 0.01, 0)  # 1% point rate
                })
        
        purchase_data = pd.DataFrame(transactions)
        print(f"✅ Generated {len(purchase_data)} transaction records")
        return purchase_data
    
    def generate_push_data(self, customer_master):
        """
        Generate push notification data (masspush_event_log_regional equivalent)
        
        Args:
            customer_master (pd.DataFrame): Customer master data
            
        Returns:
            pd.DataFrame: Push notification event data
        """
        print("🔄 Generating push notification data...")
        
        push_events = []
        
        # Generate push campaigns (last 6 months)
        campaign_dates = [
            datetime.now() - timedelta(days=np.random.randint(1, 180))
            for _ in range(50)  # 50 campaigns
        ]
        
        for customer in customer_master.itertuples():
            # Participation probability based on contact preference and mobile ownership
            if customer.contact_member_class == 'NG' or customer.mobile_owned_class == 'N':
                participation_prob = 0.1
            else:
                participation_prob = 0.7
            
            for campaign_date in campaign_dates:
                if np.random.random() < participation_prob:
                    push_content_id = f"push_content_{np.random.randint(1, 100)}"
                    
                    # Send event
                    push_events.append({
                        'hash_id': customer.hash_id,
                        'event_name': 'message_send',
                        'event_datetime': campaign_date,
                        'push_content_id': push_content_id
                    })
                    
                    # Click event (30% click rate)
                    if np.random.random() < 0.3:
                        click_time = campaign_date + timedelta(minutes=np.random.randint(1, 120))
                        push_events.append({
                            'hash_id': customer.hash_id,
                            'event_name': 'message_click',
                            'event_datetime': click_time,
                            'push_content_id': push_content_id
                        })
        
        push_data = pd.DataFrame(push_events)
        print(f"✅ Generated {len(push_data)} push event records")
        return push_data
    
    def _get_age_distribution(self):
        """Get realistic age distribution for Japanese retail customers"""
        age_ranges = range(18, 80)
        # Simulate realistic age distribution (peak in 30-50s)
        probabilities = stats.norm.pdf(age_ranges, loc=40, scale=15)
        return probabilities / probabilities.sum()
    
    def calculate_rfm_features(self, purchase_data):
        """
        Calculate RFM (Recency, Frequency, Monetary) features
        This is the foundation for advanced segmentation
        
        Args:
            purchase_data (pd.DataFrame): Purchase transaction data
            
        Returns:
            pd.DataFrame: RFM features by customer
        """
        print("🔄 Calculating RFM features...")
        
        # Reference date for recency calculation
        reference_date = datetime.now()
        
        # Calculate RFM metrics
        rfm = purchase_data.groupby('hash_id').agg({
            'transaction_date': lambda x: (reference_date - x.max()).days,  # Recency
            'hash_id': 'count',  # Frequency (transaction count)
            'sales_amount': 'sum'  # Monetary (total spent)
        }).round(2)
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Add additional behavioral metrics
        additional_metrics = purchase_data.groupby('hash_id').agg({
            'sales_amount': ['mean', 'std'],  # Average order value and variability
            'discount_amount': 'mean',  # Average discount used
            'quantity': 'mean'  # Average items per transaction
        }).round(2)
        
        additional_metrics.columns = ['AOV', 'AOV_Std', 'Avg_Discount', 'Avg_Items']
        
        # Combine all metrics
        customer_features = pd.concat([rfm, additional_metrics], axis=1)
        customer_features['AOV_Std'] = customer_features['AOV_Std'].fillna(0)
        
        print(f"✅ Calculated RFM features for {len(customer_features)} customers")
        return customer_features.reset_index()


def main():
    """
    Main execution function to generate all sample data
    """
    print("🚀 Starting Customer Data Generation Process")
    print("=" * 50)
    
    # Initialize generator
    generator = CustomerDataGenerator(n_customers=5000)  # Start with 5k customers
    
    # Generate all datasets
    customer_master = generator.generate_customer_master()
    purchase_data = generator.generate_purchase_data(customer_master)
    push_data = generator.generate_push_data(customer_master)
    rfm_features = generator.calculate_rfm_features(purchase_data)
    
    # Save data
    print("\n💾 Saving generated data...")
    customer_master.to_csv('customer_analysis/segmentation/data/customer_master.csv', index=False)
    purchase_data.to_csv('customer_analysis/segmentation/data/purchase_data.csv', index=False)
    push_data.to_csv('customer_analysis/segmentation/data/push_data.csv', index=False)
    rfm_features.to_csv('customer_analysis/segmentation/data/rfm_features.csv', index=False)
    
    # Display summary
    print("\n📊 Data Generation Summary:")
    print(f"  • Customer Master: {len(customer_master):,} records")
    print(f"  • Purchase Data: {len(purchase_data):,} transactions")
    print(f"  • Push Data: {len(push_data):,} events")
    print(f"  • RFM Features: {len(rfm_features):,} customer profiles")
    
    print("\n✅ Data generation completed successfully!")
    print("Ready for segmentation analysis 🎯")
    
    return customer_master, purchase_data, push_data, rfm_features


if __name__ == "__main__":
    main()
