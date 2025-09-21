"""
Advanced Preprocessing Pipeline for Customer Segmentation
=======================================================

NASA/Google level preprocessing with numerical stability and theoretical rigor
Implements robust scaling, dimensionality reduction, and feature engineering

Author: Data Science Team  
Purpose: Bulletproof preprocessing for clustering algorithms
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
from scipy.linalg import svd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class AdvancedPreprocessor:
    """
    Advanced preprocessing pipeline for customer segmentation
    
    Implements methods used at NASA, Google, Meta for numerical stability
    and optimal clustering performance.
    """
    
    def __init__(self, method='robust'):
        """
        Initialize preprocessor
        
        Args:
            method (str): Scaling method ('standard', 'robust', 'power')
        """
        self.method = method
        self.scalers = {}
        self.transformers = {}
        self.pca = None
        self.feature_importance = None
        
    def detect_optimal_scaling(self, data):
        """
        Automatically detect optimal scaling method based on data properties
        
        Args:
            data (pd.DataFrame): Input features
            
        Returns:
            str: Recommended scaling method
        """
        print("🔍 Analyzing data properties for optimal scaling...")
        
        # Calculate skewness and outlier percentage for each feature
        analysis = {}
        
        for col in data.columns:
            values = data[col].dropna()
            
            # Skewness
            skew = abs(stats.skew(values))
            
            # Outlier percentage (IQR method)
            Q1, Q3 = values.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = ((values < Q1 - 1.5*IQR) | (values > Q3 + 1.5*IQR)).sum()
            outlier_pct = outliers / len(values) * 100
            
            analysis[col] = {
                'skewness': skew,
                'outlier_pct': outlier_pct
            }
        
        # Decision logic (based on Google's best practices)
        avg_skew = np.mean([analysis[col]['skewness'] for col in data.columns])
        avg_outliers = np.mean([analysis[col]['outlier_pct'] for col in data.columns])
        
        print(f"Average skewness: {avg_skew:.3f}")
        print(f"Average outlier percentage: {avg_outliers:.2f}%")
        
        if avg_outliers > 10:  # High outlier presence
            recommendation = "robust"
            reason = "High outlier presence (>10%)"
        elif avg_skew > 2:  # Highly skewed
            recommendation = "power"
            reason = "High skewness (>2.0)"
        else:
            recommendation = "standard" 
            reason = "Normal distribution characteristics"
            
        print(f"💡 Recommended scaling: {recommendation} ({reason})")
        
        return recommendation, analysis
    
    def apply_scaling(self, data, method=None):
        """
        Apply scaling with the specified method
        
        Args:
            data (pd.DataFrame): Input features
            method (str): Scaling method to use
            
        Returns:
            np.ndarray: Scaled features
        """
        if method is None:
            method = self.method
            
        print(f"⚙️ Applying {method} scaling...")
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()  # Uses median and IQR, robust to outliers
        elif method == 'power':
            scaler = PowerTransformer(method='yeo-johnson')  # Handles zeros and negatives
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        scaled_data = scaler.fit_transform(data)
        
        # Store scaler for future use
        self.scalers[method] = scaler
        
        # Convert back to DataFrame with original column names
        scaled_df = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        
        print(f"✅ Scaling completed. Shape: {scaled_df.shape}")
        
        return scaled_df, scaler
    
    def reduce_dimensionality(self, data, method='pca', n_components=None):
        """
        Apply dimensionality reduction with theoretical rigor
        
        Args:
            data (pd.DataFrame): Scaled input features
            method (str): Reduction method ('pca', 'kernel_pca')
            n_components (int): Number of components to keep
            
        Returns:
            np.ndarray: Reduced dimensional data
        """
        print(f"\n🔬 Applying {method.upper()} dimensionality reduction...")
        
        if method == 'pca':
            # Determine optimal number of components if not specified
            if n_components is None:
                n_components = self._find_optimal_components(data)
            
            # Apply PCA
            pca = PCA(n_components=n_components, random_state=42)
            reduced_data = pca.fit_transform(data)
            
            # Store PCA object
            self.pca = pca
            
            # Print explained variance
            explained_var = pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            print(f"Components selected: {n_components}")
            print(f"Explained variance by component: {explained_var.round(3)}")
            print(f"Cumulative explained variance: {cumulative_var.round(3)}")
            print(f"Total variance explained: {cumulative_var[-1]:.1%}")
            
            # Analyze component loadings
            self._analyze_pca_loadings(pca, data.columns)
            
        else:
            raise NotImplementedError(f"Method {method} not implemented yet")
            
        return reduced_data, pca
    
    def _find_optimal_components(self, data, variance_threshold=0.95):
        """
        Find optimal number of PCA components using multiple criteria
        
        Args:
            data (pd.DataFrame): Input data
            variance_threshold (float): Minimum variance to explain
            
        Returns:
            int: Optimal number of components
        """
        print("🎯 Finding optimal number of components...")
        
        # Fit PCA with all components
        pca_full = PCA()
        pca_full.fit(data)
        
        explained_var = pca_full.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        # Method 1: Variance threshold
        n_variance = np.argmax(cumulative_var >= variance_threshold) + 1
        
        # Method 2: Kaiser criterion (eigenvalue > 1)
        eigenvalues = pca_full.explained_variance_
        n_kaiser = np.sum(eigenvalues > 1)
        
        # Method 3: Elbow method (second derivative)
        second_deriv = np.diff(explained_var, 2)
        n_elbow = np.argmax(second_deriv) + 2 if len(second_deriv) > 0 else n_kaiser
        
        print(f"Method 1 - Variance threshold ({variance_threshold:.0%}): {n_variance} components")
        print(f"Method 2 - Kaiser criterion (eigenvalue > 1): {n_kaiser} components")
        print(f"Method 3 - Elbow method: {n_elbow} components")
        
        # Conservative choice: minimum of methods 1 and 2
        optimal_n = min(n_variance, max(n_kaiser, 2))  # At least 2 components
        
        print(f"💡 Selected: {optimal_n} components (conservative approach)")
        
        return optimal_n
    
    def _analyze_pca_loadings(self, pca, feature_names):
        """
        Analyze PCA component loadings for interpretability
        
        Args:
            pca: Fitted PCA object
            feature_names: Original feature names
        """
        print("\n📊 PCA Component Analysis:")
        print("-" * 40)
        
        loadings = pca.components_
        
        for i in range(min(3, loadings.shape[0])):  # Show first 3 components
            print(f"\nComponent {i+1} (explains {pca.explained_variance_ratio_[i]:.1%}):")
            
            # Get loading strengths
            component_loadings = pd.Series(loadings[i], index=feature_names)
            
            # Sort by absolute loading value
            sorted_loadings = component_loadings.abs().sort_values(ascending=False)
            
            for feature in sorted_loadings.index[:3]:  # Top 3 features
                loading_val = component_loadings[feature]
                print(f"  {feature:15} | Loading: {loading_val:6.3f}")
    
    def detect_multicollinearity(self, data):
        """
        Detect multicollinearity using Variance Inflation Factor (VIF)
        
        Args:
            data (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: VIF scores
        """
        print("\n🔍 Multicollinearity Analysis (VIF):")
        print("-" * 35)
        
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            vif_data = pd.DataFrame()
            vif_data["Feature"] = data.columns
            vif_data["VIF"] = [variance_inflation_factor(data.values, i) 
                             for i in range(data.shape[1])]
            
            # Sort by VIF score
            vif_data = vif_data.sort_values('VIF', ascending=False)
            
            print(vif_data)
            
            # Interpretation
            high_vif = vif_data[vif_data['VIF'] > 10]
            if not high_vif.empty:
                print(f"\n⚠️ High multicollinearity detected (VIF > 10):")
                for _, row in high_vif.iterrows():
                    print(f"  {row['Feature']}: VIF = {row['VIF']:.2f}")
                print("💡 Consider removing highly correlated features")
            else:
                print("\n✅ No severe multicollinearity detected")
                
        except ImportError:
            print("❌ statsmodels not available. Install with: pip install statsmodels")
            vif_data = None
            
        return vif_data
    
    def create_preprocessing_pipeline(self, data, target_variance=0.95):
        """
        Complete preprocessing pipeline
        
        Args:
            data (pd.DataFrame): Raw input data
            target_variance (float): Target explained variance for PCA
            
        Returns:
            tuple: (processed_data, metadata)
        """
        print("\n" + "="*60)
        print("🔧 COMPLETE PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Basic data validation
        print("Step 1: Data validation...")
        initial_shape = data.shape
        data_clean = data.dropna()
        final_shape = data_clean.shape
        
        print(f"Initial shape: {initial_shape}")
        print(f"After removing NaN: {final_shape}")
        print(f"Rows removed: {initial_shape[0] - final_shape[0]}")
        
        # Step 2: Detect optimal scaling
        optimal_method, scaling_analysis = self.detect_optimal_scaling(data_clean)
        
        # Step 3: Apply scaling
        scaled_data, scaler = self.apply_scaling(data_clean, method=optimal_method)
        
        # Step 4: Multicollinearity check
        vif_results = self.detect_multicollinearity(scaled_data)
        
        # Step 5: Dimensionality reduction
        reduced_data, pca_obj = self.reduce_dimensionality(
            scaled_data, method='pca', n_components=None
        )
        
        # Step 6: Create final feature matrix
        feature_columns = [f'PC{i+1}' for i in range(reduced_data.shape[1])]
        final_data = pd.DataFrame(reduced_data, columns=feature_columns, 
                                index=data_clean.index)
        
        # Metadata for tracking
        metadata = {
            'original_features': list(data.columns),
            'final_features': feature_columns,
            'scaling_method': optimal_method,
            'scaling_analysis': scaling_analysis,
            'pca_explained_variance': pca_obj.explained_variance_ratio_,
            'total_variance_explained': np.sum(pca_obj.explained_variance_ratio_),
            'n_components': reduced_data.shape[1],
            'vif_results': vif_results
        }
        
        print("\n✅ Preprocessing pipeline completed successfully!")
        print(f"Final data shape: {final_data.shape}")
        print(f"Variance explained: {metadata['total_variance_explained']:.1%}")
        
        return final_data, metadata


def main():
    """
    Execute preprocessing pipeline on customer data
    """
    print("🚀 Starting Advanced Preprocessing Pipeline")
    print("="*60)
    
    # Load the clustering data from EDA
    try:
        # Assume EDA was run and clustering data is available
        import pickle
        
        # For demonstration, load RFM data directly
        rfm_data = pd.read_csv('customer_analysis/segmentation/data/rfm_features.csv')
        
        # Select features for clustering
        feature_columns = ['Recency', 'Frequency', 'Monetary', 'AOV', 'Avg_Discount', 'Avg_Items']
        clustering_features = rfm_data[feature_columns].copy()
        
        print(f"Loaded data shape: {clustering_features.shape}")
        
    except FileNotFoundError:
        print("❌ Data not found. Please run data generation and EDA first.")
        return None
    
    # Initialize preprocessor
    preprocessor = AdvancedPreprocessor()
    
    # Run complete pipeline
    processed_data, metadata = preprocessor.create_preprocessing_pipeline(clustering_features)
    
    # Save results
    processed_data.to_csv('customer_analysis/segmentation/data/processed_features.csv', index=False)
    
    # Save metadata
    import json
    with open('customer_analysis/segmentation/data/preprocessing_metadata.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metadata_json = metadata.copy()
        metadata_json['pca_explained_variance'] = metadata['pca_explained_variance'].tolist()
        if metadata['vif_results'] is not None:
            metadata_json['vif_results'] = metadata['vif_results'].to_dict()
        json.dump(metadata_json, f, indent=2)
    
    print("\n📁 Files saved:")
    print("  • processed_features.csv")
    print("  • preprocessing_metadata.json")
    
    return processed_data, metadata, preprocessor


if __name__ == "__main__":
    results = main()
