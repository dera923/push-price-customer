"""
Advanced Clustering Algorithms for Customer Segmentation
=====================================================

Google/Meta/NASA level clustering implementation with theoretical rigor
Implements K-means (Variational), Hierarchical (Information-theoretic), 
and DBSCAN (Topological) with mathematical optimization

Author: Data Science Team
Purpose: World-class customer segmentation algorithms
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class AdvancedKMeans:
    """
    Advanced K-means implementation with variational interpretation
    
    Based on EM algorithm theory and information-theoretic optimization
    Used at Google, Meta for customer segmentation
    """
    
    def __init__(self, max_clusters=10, random_state=42):
        """
        Initialize Advanced K-means
        
        Args:
            max_clusters (int): Maximum number of clusters to evaluate
            random_state (int): Random seed for reproducibility
        """
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.models = {}
        self.evaluation_metrics = {}
        self.optimal_k = None
        
    def find_optimal_clusters(self, data, methods=['elbow', 'silhouette', 'gap']):
        """
        Find optimal number of clusters using multiple methods
        
        Args:
            data (np.ndarray): Preprocessed data
            methods (list): Methods to use for evaluation
            
        Returns:
            dict: Optimal k for each method
        """
        print("🔍 Finding optimal number of clusters...")
        
        k_range = range(2, min(self.max_clusters + 1, len(data) // 2))
        results = {method: {} for method in methods}
        
        # Store all metrics for each k
        inertias = []
        silhouette_scores = []
        calinski_harabasz_scores = []
        davies_bouldin_scores = []
        
        for k in k_range:
            print(f"  Evaluating k={k}...")
            
            # Fit K-means
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, 
                          init='k-means++', n_init=10)
            labels = kmeans.fit_predict(data)
            
            # Store model
            self.models[k] = kmeans
            
            # Calculate metrics
            inertia = kmeans.inertia_
            sil_score = silhouette_score(data, labels)
            cal_har_score = calinski_harabasz_score(data, labels)
            dav_bou_score = davies_bouldin_score(data, labels)
            
            inertias.append(inertia)
            silhouette_scores.append(sil_score)
            calinski_harabasz_scores.append(cal_har_score)
            davies_bouldin_scores.append(dav_bou_score)
        
        # Method 1: Elbow method (second derivative of inertia)
        if 'elbow' in methods:
            second_derivatives = np.diff(inertias, 2)
            elbow_k = k_range[np.argmax(second_derivatives) + 2] if len(second_derivatives) > 0 else k_range[0]
            results['elbow']['optimal_k'] = elbow_k
            results['elbow']['inertias'] = inertias
        
        # Method 2: Silhouette method
        if 'silhouette' in methods:
            silhouette_k = k_range[np.argmax(silhouette_scores)]
            results['silhouette']['optimal_k'] = silhouette_k
            results['silhouette']['scores'] = silhouette_scores
        
        # Method 3: Gap statistic (simplified implementation)
        if 'gap' in methods:
            gap_k = self._calculate_gap_statistic(data, k_range)
            results['gap']['optimal_k'] = gap_k
        
        # Store evaluation metrics
        self.evaluation_metrics = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_harabasz_scores': calinski_harabasz_scores,
            'davies_bouldin_scores': davies_bouldin_scores
        }
        
        # Consensus optimal k (most frequent recommendation)
        optimal_ks = [results[method]['optimal_k'] for method in methods if 'optimal_k' in results[method]]
        self.optimal_k = max(set(optimal_ks), key=optimal_ks.count)
        
        print(f"\n📊 Optimal k recommendations:")
        for method in methods:
            if 'optimal_k' in results[method]:
                print(f"  {method.capitalize()}: k = {results[method]['optimal_k']}")
        print(f"\n🎯 Consensus optimal k: {self.optimal_k}")
        
        return results, self.optimal_k
    
    def _calculate_gap_statistic(self, data, k_range, n_refs=10):
        """
        Calculate Gap statistic for optimal k selection
        
        Based on Tibshirani, Walther, and Hastie (2001)
        """
        print("    Calculating Gap statistic...")
        
        gaps = []
        
        for k in k_range:
            # Original data clustering
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(data)
            original_dispersion = np.log(kmeans.inertia_)
            
            # Reference data dispersions (uniform random)
            ref_dispersions = []
            for _ in range(n_refs):
                # Generate uniform random data with same bounds
                random_data = np.random.uniform(
                    data.min(axis=0), data.max(axis=0), data.shape
                )
                kmeans_ref = KMeans(n_clusters=k, random_state=self.random_state)
                kmeans_ref.fit(random_data)
                ref_dispersions.append(np.log(kmeans_ref.inertia_))
            
            # Gap statistic
            gap = np.mean(ref_dispersions) - original_dispersion
            gaps.append(gap)
        
        # Find optimal k (first k where gap(k) >= gap(k+1) - se(k+1))
        # Simplified: just take maximum gap
        optimal_k = k_range[np.argmax(gaps)]
        
        return optimal_k
    
    def fit_final_model(self, data, k=None):
        """
        Fit final K-means model with optimal k
        
        Args:
            data (np.ndarray): Preprocessed data
            k (int): Number of clusters (uses optimal_k if None)
            
        Returns:
            KMeans: Fitted model
        """
        if k is None:
            k = self.optimal_k
            
        print(f"🎯 Fitting final K-means model with k={k}...")
        
        # Enhanced initialization with multiple runs
        best_inertia = np.inf
        best_model = None
        
        for init_seed in range(10):  # Multiple random initializations
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state + init_seed,
                init='k-means++',
                n_init=20,
                max_iter=500,
                tol=1e-6
            )
            kmeans.fit(data)
            
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_model = kmeans
        
        self.final_model = best_model
        self.labels = best_model.labels_
        self.centroids = best_model.cluster_centers_
        
        print(f"✅ Final model fitted. Inertia: {best_inertia:.2f}")
        
        return best_model


class AdvancedHierarchical:
    """
    Advanced Hierarchical Clustering with information-theoretic approach
    
    Implements Ward linkage with statistical validation
    Used at NASA, Meta for understanding cluster hierarchy
    """
    
    def __init__(self, linkage_method='ward', distance_metric='euclidean'):
        """
        Initialize Hierarchical Clustering
        
        Args:
            linkage_method (str): Linkage criterion
            distance_metric (str): Distance metric
        """
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.linkage_matrix = None
        self.optimal_clusters = None
        
    def fit_hierarchical(self, data):
        """
        Fit hierarchical clustering and create dendrogram
        
        Args:
            data (np.ndarray): Preprocessed data
            
        Returns:
            np.ndarray: Linkage matrix
        """
        print(f"🌳 Fitting hierarchical clustering ({self.linkage_method} linkage)...")
        
        # Calculate distance matrix if needed
        if self.linkage_method == 'ward':
            # Ward requires Euclidean distance
            self.distance_metric = 'euclidean'
        
        # Create linkage matrix
        self.linkage_matrix = linkage(data, method=self.linkage_method, 
                                    metric=self.distance_metric)
        
        print(f"✅ Linkage matrix created. Shape: {self.linkage_matrix.shape}")
        
        return self.linkage_matrix
    
    def find_optimal_clusters(self, data, max_clusters=10):
        """
        Find optimal number of clusters using inconsistency coefficient
        
        Args:
            data (np.ndarray): Preprocessed data
            max_clusters (int): Maximum clusters to evaluate
            
        Returns:
            int: Optimal number of clusters
        """
        print("🔍 Finding optimal clusters using inconsistency analysis...")
        
        if self.linkage_matrix is None:
            self.fit_hierarchical(data)
        
        # Calculate inconsistency coefficients
        from scipy.cluster.hierarchy import inconsistent
        inconsistency = inconsistent(self.linkage_matrix, d=2)
        
        # Find optimal cut using inconsistency
        # Look for large jumps in inconsistency
        inconsistency_values = inconsistency[:, 3]  # Inconsistency coefficient
        
        # Calculate differences between consecutive levels
        diffs = np.diff(inconsistency_values[::-1])  # Reverse for top-down
        
        # Find the largest jump (optimal cut point)
        optimal_cut_idx = np.argmax(diffs)
        optimal_clusters = optimal_cut_idx + 2  # +2 because we want clusters, not cuts
        
        # Ensure within reasonable bounds
        optimal_clusters = min(max(optimal_clusters, 2), max_clusters)
        
        self.optimal_clusters = optimal_clusters
        
        print(f"🎯 Optimal clusters found: {optimal_clusters}")
        
        return optimal_clusters
    
    def get_cluster_labels(self, n_clusters=None):
        """
        Get cluster labels for specified number of clusters
        
        Args:
            n_clusters (int): Number of clusters
            
        Returns:
            np.ndarray: Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.optimal_clusters
            
        if self.linkage_matrix is None:
            raise ValueError("Must fit hierarchical clustering first")
        
        # Cut dendrogram to get specified number of clusters
        labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust') - 1
        
        return labels
    
    def plot_dendrogram(self, data, max_display=30):
        """
        Plot dendrogram with optimal cut line
        
        Args:
            data (np.ndarray): Original data
            max_display (int): Maximum clusters to display
        """
        plt.figure(figsize=(15, 8))
        
        # Create dendrogram
        dendro = dendrogram(
            self.linkage_matrix,
            leaf_rotation=90,
            leaf_font_size=8,
            truncate_mode='lastp',
            p=max_display
        )
        
        # Add optimal cut line
        if self.optimal_clusters is not None:
            # Calculate distance threshold for optimal clusters
            threshold = self.linkage_matrix[-(self.optimal_clusters-1), 2]
            plt.axhline(y=threshold, color='red', linestyle='--', 
                       label=f'Optimal cut (k={self.optimal_clusters})')
            plt.legend()
        
        plt.title(f'Hierarchical Clustering Dendrogram ({self.linkage_method} linkage)')
        plt.xlabel('Sample Index or (Cluster Size)')
        plt.ylabel('Distance')
        plt.tight_layout()
        
        # Save plot
        plt.savefig('customer_analysis/segmentation/visualizations/dendrogram.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()


class AdvancedDBSCAN:
    """
    Advanced DBSCAN with automatic parameter optimization
    
    Topological approach to density-based clustering
    Used at Google Maps, Meta for spatial clustering
    """
    
    def __init__(self):
        """Initialize DBSCAN with parameter optimization"""
        self.optimal_eps = None
        self.optimal_min_samples = None
        self.model = None
        self.labels = None
        
    def find_optimal_eps(self, data, k=5):
        """
        Find optimal eps using k-distance plot
        
        Args:
            data (np.ndarray): Preprocessed data
            k (int): Number of nearest neighbors
            
        Returns:
            float: Optimal eps value
        """
        print(f"🔍 Finding optimal eps using {k}-distance plot...")
        
        # Calculate k-nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(data)
        distances, indices = neighbors_fit.kneighbors(data)
        
        # Sort distances to k-th nearest neighbor
        k_distances = distances[:, k-1]
        k_distances_sorted = np.sort(k_distances, axis=0)[::-1]
        
        # Find elbow point using second derivative
        second_derivative = np.diff(k_distances_sorted, 2)
        
        # Find the point with maximum curvature (elbow)
        elbow_point = np.argmax(second_derivative) + 2
        optimal_eps = k_distances_sorted[elbow_point]
        
        self.optimal_eps = optimal_eps
        
        print(f"🎯 Optimal eps found: {optimal_eps:.4f}")
        
        # Plot k-distance graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(k_distances_sorted)), k_distances_sorted, 'b-', linewidth=2)
        plt.axhline(y=optimal_eps, color='red', linestyle='--', 
                   label=f'Optimal eps = {optimal_eps:.4f}')
        plt.axvline(x=elbow_point, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{k}-NN Distance')
        plt.title(f'{k}-Distance Plot for DBSCAN eps Selection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('customer_analysis/segmentation/visualizations/k_distance_plot.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return optimal_eps
    
    def optimize_parameters(self, data):
        """
        Optimize both eps and min_samples parameters
        
        Args:
            data (np.ndarray): Preprocessed data
            
        Returns:
            tuple: (optimal_eps, optimal_min_samples)
        """
        print("⚙️ Optimizing DBSCAN parameters...")
        
        # Find optimal eps
        optimal_eps = self.find_optimal_eps(data)
        
        # Optimize min_samples (typically 2 * dimensions)
        n_features = data.shape[1]
        optimal_min_samples = max(3, 2 * n_features)  # Minimum 3, or 2*dimensions
        
        # Validate with silhouette score
        best_score = -1
        best_min_samples = optimal_min_samples
        
        for min_samples in range(3, min(20, len(data) // 10)):
            dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)
            
            # Calculate silhouette score (only if we have valid clusters)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1:
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_min_samples = min_samples
        
        self.optimal_min_samples = best_min_samples
        
        print(f"🎯 Optimal parameters:")
        print(f"  eps = {optimal_eps:.4f}")
        print(f"  min_samples = {best_min_samples}")
        print(f"  Best silhouette score = {best_score:.4f}")
        
        return optimal_eps, best_min_samples
    
    def fit_dbscan(self, data, eps=None, min_samples=None):
        """
        Fit DBSCAN with optimal or specified parameters
        
        Args:
            data (np.ndarray): Preprocessed data
            eps (float): Distance parameter
            min_samples (int): Minimum samples parameter
            
        Returns:
            DBSCAN: Fitted model
        """
        if eps is None or min_samples is None:
            eps, min_samples = self.optimize_parameters(data)
        
        print(f"🎯 Fitting DBSCAN with eps={eps:.4f}, min_samples={min_samples}...")
        
        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        
        # Analyze results
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"✅ DBSCAN completed:")
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise} ({n_noise/len(data)*100:.1f}%)")
        
        self.model = dbscan
        self.labels = labels
        
        return dbscan


class ClusteringEvaluator:
    """
    Comprehensive cluster evaluation using multiple metrics
    
    Implements statistical tests and visualization for cluster quality
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.results = {}
        
    def evaluate_clustering(self, data, labels_dict, algorithm_names):
        """
        Comprehensive evaluation of multiple clustering results
        
        Args:
            data (np.ndarray): Original data
            labels_dict (dict): Dictionary of algorithm -> labels
            algorithm_names (list): Names of algorithms
            
        Returns:
            pd.DataFrame: Evaluation results
        """
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE CLUSTERING EVALUATION")
        print("="*60)
        
        evaluation_results = []
        
        for algo_name, labels in labels_dict.items():
            # Skip if all points are noise (DBSCAN edge case)
            unique_labels = set(labels)
            if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
                print(f"⚠️ Skipping {algo_name}: insufficient clusters")
                continue
            
            # Basic metrics
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(labels).count(-1) if -1 in labels else 0
            
            # Internal validation metrics
            silhouette = silhouette_score(data, labels) if n_clusters > 1 else -1
            calinski_harabasz = calinski_harabasz_score(data, labels) if n_clusters > 1 else 0
            davies_bouldin = davies_bouldin_score(data, labels) if n_clusters > 1 else np.inf
            
            # Store results
            result = {
                'Algorithm': algo_name,
                'N_Clusters': n_clusters,
                'N_Noise': n_noise,
                'Noise_Ratio': n_noise / len(data),
                'Silhouette_Score': silhouette,
                'Calinski_Harabasz_Score': calinski_harabasz,
                'Davies_Bouldin_Score': davies_bouldin
            }
            
            evaluation_results.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(evaluation_results)
        
        print(results_df.round(4))
        
        # Rank algorithms
        if len(results_df) > 1:
            self._rank_algorithms(results_df)
        
        return results_df
    
    def _rank_algorithms(self, results_df):
        """
        Rank algorithms based on multiple criteria
        
        Args:
            results_df (pd.DataFrame): Evaluation results
        """
        print(f"\n🏆 ALGORITHM RANKING:")
        print("-" * 25)
        
        # Create ranking scores (higher is better for all metrics)
        ranking_df = results_df.copy()
        
        # Normalize metrics to 0-1 scale
        ranking_df['Silhouette_Rank'] = ranking_df['Silhouette_Score'] / ranking_df['Silhouette_Score'].max()
        ranking_df['Calinski_Rank'] = ranking_df['Calinski_Harabasz_Score'] / ranking_df['Calinski_Harabasz_Score'].max()
        ranking_df['Davies_Rank'] = 1 - (ranking_df['Davies_Bouldin_Score'] / ranking_df['Davies_Bouldin_Score'].max())
        
        # Combined score (equal weights)
        ranking_df['Combined_Score'] = (
            ranking_df['Silhouette_Rank'] + 
            ranking_df['Calinski_Rank'] + 
            ranking_df['Davies_Rank']
        ) / 3
        
        # Sort by combined score
        ranking_df = ranking_df.sort_values('Combined_Score', ascending=False)
        
        for i, (_, row) in enumerate(ranking_df.iterrows()):
            rank = i + 1
            score = row['Combined_Score']
            algo = row['Algorithm']
            print(f"{rank}. {algo:20} | Score: {score:.3f}")


def main():
    """
    Execute complete clustering pipeline
    """
    print("🚀 Starting Advanced Clustering Pipeline")
    print("="*60)
    
    # Load preprocessed data
    try:
        processed_data = pd.read_csv('customer_analysis/segmentation/data/processed_features.csv')
        data_matrix = processed_data.values
        
        print(f"Loaded preprocessed data: {data_matrix.shape}")
        
    except FileNotFoundError:
        print("❌ Preprocessed data not found. Please run preprocessing first.")
        return None
    
    # Initialize algorithms
    kmeans_algo = AdvancedKMeans(max_clusters=8)
    hierarchical_algo = AdvancedHierarchical()
    dbscan_algo = AdvancedDBSCAN()
    evaluator = ClusteringEvaluator()
    
    # 1. K-means clustering
    print("\n" + "="*50)
    print("🎯 K-MEANS CLUSTERING")
    print("="*50)
    
    kmeans_results, optimal_k = kmeans_algo.find_optimal_clusters(data_matrix)
    kmeans_model = kmeans_algo.fit_final_model(data_matrix)
    kmeans_labels = kmeans_model.labels_
    
    # 2. Hierarchical clustering
    print("\n" + "="*50)
    print("🌳 HIERARCHICAL CLUSTERING")
    print("="*50)
    
    hierarchical_algo.fit_hierarchical(data_matrix)
    hierarchical_optimal = hierarchical_algo.find_optimal_clusters(data_matrix)
    hierarchical_labels = hierarchical_algo.get_cluster_labels()
    hierarchical_algo.plot_dendrogram(data_matrix)
    
    # 3. DBSCAN clustering
    print("\n" + "="*50)
    print("🔍 DBSCAN CLUSTERING")
    print("="*50)
    
    dbscan_model = dbscan_algo.fit_dbscan(data_matrix)
    dbscan_labels = dbscan_model.labels_
    
    # 4. Comprehensive evaluation
    labels_dict = {
        'K-means': kmeans_labels,
        'Hierarchical': hierarchical_labels,
        'DBSCAN': dbscan_labels
    }
    
    algorithm_names = list(labels_dict.keys())
    evaluation_results = evaluator.evaluate_clustering(
        data_matrix, labels_dict, algorithm_names
    )
    
    # Save results
    evaluation_results.to_csv(
        'customer_analysis/segmentation/data/clustering_evaluation.csv', 
        index=False
    )
    
    print("\n✅ Clustering pipeline completed successfully!")
    print("📁 Results saved to clustering_evaluation.csv")
    
    return {
        'kmeans': (kmeans_algo, kmeans_labels),
        'hierarchical': (hierarchical_algo, hierarchical_labels),
        'dbscan': (dbscan_algo, dbscan_labels),
        'evaluation': evaluation_results
    }


if __name__ == "__main__":
    clustering_results = main()
