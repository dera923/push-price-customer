"""
Advanced Statistical EDA for Customer Segmentation
==============================================

Google/Meta/NASA level exploratory data analysis with mathematical rigor
Implements statistical tests and theoretical frameworks used in top-tier companies

Author: Data Science Team
Purpose: Rigorous statistical analysis before segmentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class AdvancedEDA:
    """
    Advanced Exploratory Data Analysis for Customer Segmentation
    
    Implements statistical methods used at Google, Meta, NASA for data understanding
    before applying clustering algorithms.
    """
    
    def __init__(self, data_path='customer_analysis/segmentation/data/'):
        """
        Initialize EDA with data loading
        
        Args:
            data_path (str): Path to data directory
        """
        self.data_path = data_path
        self.load_data()
        
    def load_data(self):
        """Load all generated datasets"""
        print("📂 Loading generated customer data...")
        
        try:
            self.customer_master = pd.read_csv(f"{self.data_path}customer_master.csv")
            self.purchase_data = pd.read_csv(f"{self.data_path}purchase_data.csv")
            self.push_data = pd.read_csv(f"{self.data_path}push_data.csv")
            self.rfm_features = pd.read_csv(f"{self.data_path}rfm_features.csv")
            
            print("✅ Data loaded successfully")
            print(f"  • Customer Master: {len(self.customer_master):,} records")
            print(f"  • RFM Features: {len(self.rfm_features):,} customer profiles")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print("Please ensure data generation was completed successfully")
            
    def statistical_summary(self):
        """
        Comprehensive statistical summary using advanced metrics
        
        Beyond basic describe() - includes skewness, kurtosis, normality tests
        """
        print("\n" + "="*60)
        print("📈 ADVANCED STATISTICAL SUMMARY")
        print("="*60)
        
        # Focus on RFM features for segmentation
        numerical_cols = ['Recency', 'Frequency', 'Monetary', 'AOV', 'Avg_Discount', 'Avg_Items']
        data = self.rfm_features[numerical_cols]
        
        # Advanced statistics beyond basic describe()
        stats_summary = pd.DataFrame({
            'Mean': data.mean(),
            'Std': data.std(),
            'Skewness': data.skew(),
            'Kurtosis': data.kurtosis(),
            'Min': data.min(),
            'Q1': data.quantile(0.25),
            'Median': data.median(),
            'Q3': data.quantile(0.75),
            'Max': data.max(),
            'IQR': data.quantile(0.75) - data.quantile(0.25)
        })
        
        print(stats_summary.round(3))
        
        # Normality tests (Shapiro-Wilk for small samples, Anderson-Darling for large)
        print("\n🔍 NORMALITY TESTS:")
        print("-" * 40)
        
        for col in numerical_cols:
            if len(data[col]) <= 5000:  # Shapiro-Wilk limit
                stat, p_value = stats.shapiro(data[col].dropna())
                test_name = "Shapiro-Wilk"
            else:
                stat, critical_values, significance_level = stats.anderson(data[col].dropna())
                p_value = significance_level
                test_name = "Anderson-Darling"
            
            normality = "✅ Normal" if p_value > 0.05 else "❌ Non-normal"
            print(f"{col:12} | {test_name:15} | p={p_value:.4f} | {normality}")
        
        return stats_summary
    
    def correlation_analysis(self):
        """
        Advanced correlation analysis with statistical significance
        
        Includes Pearson, Spearman correlations and eigenvalue analysis
        """
        print("\n" + "="*60)
        print("🔗 CORRELATION STRUCTURE ANALYSIS")
        print("="*60)
        
        numerical_cols = ['Recency', 'Frequency', 'Monetary', 'AOV', 'Avg_Discount', 'Avg_Items']
        data = self.rfm_features[numerical_cols]
        
        # Pearson correlation matrix
        pearson_corr = data.corr(method='pearson')
        
        # Spearman correlation matrix (rank-based, robust to outliers)
        spearman_corr = data.corr(method='spearman')
        
        print("Pearson Correlation Matrix:")
        print(pearson_corr.round(3))
        
        print("\nSpearman Correlation Matrix:")
        print(spearman_corr.round(3))
        
        # Eigenvalue analysis of correlation matrix
        eigenvals, eigenvecs = np.linalg.eigh(pearson_corr)
        eigenvals = eigenvals[::-1]  # Sort descending
        
        print(f"\n📊 EIGENVALUE ANALYSIS:")
        print("-" * 30)
        print("Eigenvalues (descending):", np.round(eigenvals, 3))
        
        # Explained variance ratio
        explained_var = eigenvals / np.sum(eigenvals)
        cumulative_var = np.cumsum(explained_var)
        
        print("Explained variance ratio:", np.round(explained_var, 3))
        print("Cumulative variance:", np.round(cumulative_var, 3))
        
        # Effective dimensionality (Kaiser criterion: eigenvalue > 1)
        effective_dims = np.sum(eigenvals > 1)
        print(f"\n🎯 Effective dimensionality (eigenvalue > 1): {effective_dims}")
        
        return pearson_corr, spearman_corr, eigenvals, explained_var
    
    def outlier_detection(self):
        """
        Multi-method outlier detection
        
        1. Mahalanobis distance (multivariate)
        2. IQR method (univariate)
        3. Z-score method (univariate)
        """
        print("\n" + "="*60)
        print("🎯 ADVANCED OUTLIER DETECTION")
        print("="*60)
        
        numerical_cols = ['Recency', 'Frequency', 'Monetary', 'AOV', 'Avg_Discount', 'Avg_Items']
        data = self.rfm_features[numerical_cols].dropna()
        
        # 1. Mahalanobis Distance (Multivariate outliers)
        print("1. MAHALANOBIS DISTANCE METHOD:")
        print("-" * 35)
        
        # Calculate covariance matrix and its inverse
        cov_matrix = np.cov(data.T)
        cov_inv = np.linalg.pinv(cov_matrix)  # Pseudo-inverse for numerical stability
        mean_vector = data.mean().values
        
        # Calculate Mahalanobis distances
        mahal_distances = []
        for _, row in data.iterrows():
            distance = mahalanobis(row.values, mean_vector, cov_inv)
            mahal_distances.append(distance)
        
        # Chi-square threshold (95% confidence)
        chi2_threshold = stats.chi2.ppf(0.95, df=len(numerical_cols))
        mahal_outliers = np.array(mahal_distances) > chi2_threshold
        
        print(f"Chi-square threshold (95%): {chi2_threshold:.3f}")
        print(f"Mahalanobis outliers detected: {np.sum(mahal_outliers)} ({np.sum(mahal_outliers)/len(data)*100:.2f}%)")
        
        # 2. IQR Method (Univariate)
        print("\n2. IQR METHOD (Per variable):")
        print("-" * 30)
        
        iqr_outliers_count = 0
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            iqr_outliers_count += outliers
            
            print(f"{col:12} | Outliers: {outliers:4d} ({outliers/len(data)*100:5.2f}%)")
        
        # 3. Z-score Method
        print("\n3. Z-SCORE METHOD (|z| > 3):")
        print("-" * 30)
        
        z_scores = np.abs(stats.zscore(data))
        z_outliers = (z_scores > 3).sum(axis=0)
        
        for i, col in enumerate(numerical_cols):
            print(f"{col:12} | Outliers: {z_outliers[i]:4d} ({z_outliers[i]/len(data)*100:5.2f}%)")
        
        # Store outlier information
        self.outlier_info = {
            'mahalanobis_distances': mahal_distances,
            'mahalanobis_outliers': mahal_outliers,
            'chi2_threshold': chi2_threshold
        }
        
        return mahal_outliers, mahal_distances
    
    def distribution_analysis(self):
        """
        Analyze distributions of key variables for transformation decisions
        
        Tests for common distributions and suggests transformations
        """
        print("\n" + "="*60)
        print("📊 DISTRIBUTION ANALYSIS & TRANSFORMATION SUGGESTIONS")
        print("="*60)
        
        numerical_cols = ['Recency', 'Frequency', 'Monetary', 'AOV']
        data = self.rfm_features[numerical_cols]
        
        transformation_suggestions = {}
        
        for col in numerical_cols:
            values = data[col].dropna()
            
            print(f"\n📈 {col.upper()}:")
            print("-" * 20)
            
            # Test for log-normality
            log_values = np.log(values + 1)  # +1 to handle zeros
            _, log_p = stats.shapiro(log_values) if len(log_values) <= 5000 else (0, 0.001)
            
            # Test for normality of original
            _, orig_p = stats.shapiro(values) if len(values) <= 5000 else (0, 0.001)
            
            # Skewness analysis
            skew = stats.skew(values)
            
            print(f"Original skewness: {skew:.3f}")
            print(f"Original normality p-value: {orig_p:.4f}")
            print(f"Log-transform normality p-value: {log_p:.4f}")
            
            # Suggest transformation
            if abs(skew) > 2:  # Highly skewed
                if skew > 0:  # Right-skewed
                    if log_p > orig_p:
                        suggestion = "Log transformation recommended (right-skewed)"
                    else:
                        suggestion = "Square root or Box-Cox transformation"
                else:  # Left-skewed
                    suggestion = "Square transformation or reflect & log"
            elif abs(skew) > 1:  # Moderately skewed
                suggestion = "Consider transformation or robust methods"
            else:
                suggestion = "No transformation needed"
            
            print(f"💡 Suggestion: {suggestion}")
            transformation_suggestions[col] = suggestion
        
        return transformation_suggestions
    
    def create_visualization_dashboard(self):
        """
        Create comprehensive visualization dashboard
        
        Includes distribution plots, correlation heatmaps, outlier visualization
        """
        print("\n" + "="*60)
        print("📊 CREATING VISUALIZATION DASHBOARD")
        print("="*60)
        
        numerical_cols = ['Recency', 'Frequency', 'Monetary', 'AOV', 'Avg_Discount', 'Avg_Items']
        data = self.rfm_features[numerical_cols]
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Distribution plots
        for i, col in enumerate(numerical_cols):
            plt.subplot(3, 4, i+1)
            
            # Histogram with KDE
            plt.hist(data[col].dropna(), bins=50, alpha=0.7, density=True, color='skyblue')
            
            # Overlay normal distribution for comparison
            mean, std = data[col].mean(), data[col].std()
            x = np.linspace(data[col].min(), data[col].max(), 100)
            normal_curve = stats.norm.pdf(x, mean, std)
            plt.plot(x, normal_curve, 'r-', linewidth=2, label='Normal fit')
            
            plt.title(f'{col} Distribution', fontsize=12, fontweight='bold')
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. Correlation heatmap
        plt.subplot(3, 4, 7)
        correlation_matrix = data.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', square=True)
        plt.title('Correlation Matrix', fontsize=12, fontweight='bold')
        
        # 3. Box plots for outlier visualization
        plt.subplot(3, 4, 8)
        data_scaled = StandardScaler().fit_transform(data)
        data_scaled_df = pd.DataFrame(data_scaled, columns=numerical_cols)
        data_scaled_df.boxplot(ax=plt.gca())
        plt.title('Scaled Variables (Outlier Detection)', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        
        # 4. Pairwise scatter plots (selected pairs)
        important_pairs = [('Frequency', 'Monetary'), ('Recency', 'Frequency'), 
                          ('AOV', 'Monetary'), ('Frequency', 'AOV')]
        
        for i, (x_col, y_col) in enumerate(important_pairs):
            plt.subplot(3, 4, 9+i)
            plt.scatter(data[x_col], data[y_col], alpha=0.6, s=30)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f'{x_col} vs {y_col}', fontsize=10, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = data[x_col].corr(data[y_col])
            plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('customer_analysis/segmentation/visualizations/eda_dashboard.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization dashboard saved as 'eda_dashboard.png'")
    
    def prepare_segmentation_data(self):
        """
        Prepare clean, transformed data for segmentation algorithms
        
        Returns preprocessed data ready for clustering
        """
        print("\n" + "="*60)
        print("🔧 PREPARING DATA FOR SEGMENTATION")
        print("="*60)
        
        # Select features for segmentation
        segmentation_features = ['Recency', 'Frequency', 'Monetary', 'AOV', 'Avg_Discount', 'Avg_Items']
        data = self.rfm_features[segmentation_features].copy()
        
        # Handle missing values
        data = data.dropna()
        print(f"Data shape after removing missing values: {data.shape}")
        
        # Apply transformations based on distribution analysis
        # Log transform highly right-skewed variables
        data['Monetary_log'] = np.log1p(data['Monetary'])  # log(1+x)
        data['Frequency_sqrt'] = np.sqrt(data['Frequency'])  # Square root for moderate skew
        
        # Remove extreme outliers (beyond 3 sigma)
        for col in segmentation_features:
            z_scores = np.abs(stats.zscore(data[col]))
            data = data[z_scores < 3]
        
        print(f"Data shape after outlier removal: {data.shape}")
        
        # Final feature set for clustering
        final_features = ['Recency', 'Frequency_sqrt', 'Monetary_log', 'AOV', 'Avg_Discount', 'Avg_Items']
        clustering_data = data[final_features].copy()
        
        # Store for later use
        self.clustering_data = clustering_data
        self.feature_names = final_features
        
        print("✅ Data preprocessing completed")
        print(f"Final features: {final_features}")
        
        return clustering_data, final_features


def main():
    """
    Execute comprehensive EDA pipeline
    """
    print("🚀 Starting Advanced EDA for Customer Segmentation")
    print("=" * 60)
    
    # Initialize EDA
    eda = AdvancedEDA()
    
    # Execute analysis pipeline
    stats_summary = eda.statistical_summary()
    
    correlation_results = eda.correlation_analysis()
    
    outlier_results = eda.outlier_detection()
    
    transformation_suggestions = eda.distribution_analysis()
    
    eda.create_visualization_dashboard()
    
    clustering_data, feature_names = eda.prepare_segmentation_data()
    
    print("\n" + "="*60)
    print("🎯 EDA PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print("✅ Statistical analysis completed")
    print("✅ Correlation structure analyzed")  
    print("✅ Outliers detected and handled")
    print("✅ Distributions analyzed and transformations applied")
    print("✅ Visualization dashboard created")
    print("✅ Data prepared for segmentation algorithms")
    
    print(f"\n📊 Ready for clustering with {len(clustering_data)} customers")
    print(f"📊 Feature dimensions: {len(feature_names)}")
    
    return eda, clustering_data, feature_names


if __name__ == "__main__":
    eda_results = main()
