import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN, KMeans
import hdbscan
from datetime import datetime

class SoundCloudVisualizer:
    def __init__(self, data_dir="./data"):
        """
        Initialize the SoundCloud visualizer.
        
        Args:
            data_dir: Directory containing the scraped data
        """
        self.data_dir = data_dir
        self.seeds_file = os.path.join(data_dir, "seeds.csv")
        self.remaining_seeds_file = os.path.join(data_dir, "remaining_seeds.csv")
        self.matrix_file = os.path.join(data_dir, "sparse_matrix.npz")
        
        # Create results directory if it doesn't exist
        self.results_dir = os.path.join(data_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self._load_data()
    
    def _load_data(self):
        """Load data from files."""
        # Check if files exist
        if not os.path.exists(self.matrix_file) or not os.path.exists(self.seeds_file) or not os.path.exists(self.remaining_seeds_file):
            raise FileNotFoundError("Required data files not found. Run the scraper first.")
        
        # Load sparse matrix
        self.sparse_matrix = scipy.sparse.load_npz(self.matrix_file)
        
        # Load seeds and remaining_seeds
        self.seeds = pd.read_csv(self.seeds_file)
        self.remaining_seeds = pd.read_csv(self.remaining_seeds_file)
        
        # Filter remaining_seeds to include only processed accounts
        self.remaining_seeds = self.remaining_seeds.iloc[:self.sparse_matrix.shape[0]]
        
        print(f"Loaded data: {self.sparse_matrix.shape[0]} accounts processed, {self.seeds.shape[0]} total accounts in network")
    
    def preprocess_data(self, min_interactions=10, max_dominance=0.4):
        """
        Preprocess data for visualization.
        
        Args:
            min_interactions: Minimum sum of interactions for an account to be included
            max_dominance: Maximum percentage of interactions that can go to a single account
        
        Returns:
            Preprocessed sparse matrix and corresponding account data
        """
        # Convert to CSR for row operations
        matrix = self.sparse_matrix.tocsr()
        
        # Filter accounts with sufficient interactions
        row_sums = np.array(matrix.sum(axis=1)).flatten()
        active_mask = row_sums >= min_interactions
        
        if not np.any(active_mask):
            raise ValueError("No accounts meet the minimum interaction threshold. Try lowering min_interactions.")
        
        active_matrix = matrix[active_mask]
        active_accounts = self.remaining_seeds[active_mask].reset_index(drop=True)
        
        print(f"Filtered to {active_matrix.shape[0]} accounts with at least {min_interactions} interactions")
        
        # Normalize rows by sum to get percentage of interactions
        row_sums = np.array(active_matrix.sum(axis=1)).flatten()
        row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
        
        # Create a diagonal matrix of reciprocal sums
        recip_sums = scipy.sparse.diags(1.0 / row_sums)
        
        # Multiply to normalize
        normalized_matrix = recip_sums @ active_matrix
        
        # Find maximum percentage per row
        max_values = normalized_matrix.max(axis=1).toarray().flatten()
        
        # Filter accounts that don't have a dominant interaction
        balanced_mask = max_values < max_dominance
        
        if not np.any(balanced_mask):
            raise ValueError("No accounts meet the dominance threshold. Try increasing max_dominance.")
        
        balanced_matrix = normalized_matrix[balanced_mask]
        balanced_accounts = active_accounts[balanced_mask].reset_index(drop=True)
        
        print(f"Filtered to {balanced_matrix.shape[0]} accounts with balanced interaction patterns")
        
        # Categorize accounts by generation based on emerge_date
        balanced_accounts = self.categorize_by_generation(balanced_accounts)
        
        return balanced_matrix, balanced_accounts
    
    def categorize_by_generation(self, account_data):
        """
        Categorize accounts by generation based on emerge_date.
        
        Args:
            account_data: DataFrame containing account data with emerge_date
        
        Returns:
            Updated DataFrame with generation column
        """
        # Create a copy to avoid modifying original
        result = account_data.copy()
        
        # Initialize generation column with default value
        result['generation'] = -1
        
        # Define generation time boundaries
        boundaries = [
            ('01/01/2010', 0),  # Before 2010
            ('01/01/2013', 1),  # 2010-2012
            ('01/01/2016', 2),  # 2013-2015
            ('01/01/2019', 3),  # 2016-2018
            ('01/01/2022', 4),  # 2019-2021
            ('01/01/2100', 5)   # 2022 onwards
        ]
        
        # Assign generations
        for i, row in result.iterrows():
            if row['emerge_date'] == 'unknown':
                continue
                
            try:
                emerge_date = pd.to_datetime(row['emerge_date'], errors='coerce')
                if pd.isna(emerge_date):
                    continue
                    
                for date_str, gen in boundaries:
                    boundary_date = pd.to_datetime(date_str)
                    if emerge_date < boundary_date:
                        result.loc[i, 'generation'] = gen
                        break
            except:
                # Skip if date parsing fails
                pass
        
        return result
    
    def reduce_dimensions(self, matrix, method='tsne', n_components=2, random_state=42):
        """
        Reduce the dimensionality of the interaction matrix.
        
        Args:
            matrix: Sparse matrix of interactions
            method: Dimensionality reduction method ('svd', 'tsne', or 'umap')
            n_components: Number of components to reduce to
            random_state: Random seed for reproducibility
            
        Returns:
            Reduced-dimensional representation
        """
        # First use SVD to reduce to intermediate dimensions (speeds up t-SNE/UMAP)
        n_intermediate = min(100, matrix.shape[1] - 1) if method != 'svd' else n_components
        
        print(f"Reducing dimensions with {method}...")
        
        # SVD for initial dimensionality reduction
        svd = TruncatedSVD(n_components=n_intermediate, random_state=random_state)
        intermediate = svd.fit_transform(matrix)
        
        print(f"Explained variance with SVD: {svd.explained_variance_ratio_.sum():.4f}")
        
        # Return SVD result directly if requested
        if method == 'svd':
            return intermediate
        
        # Further reduce with t-SNE or UMAP
        if method == 'tsne':
            tsne = TSNE(
                n_components=n_components,
                learning_rate='auto',
                init='pca',
                random_state=random_state
            )
            return tsne.fit_transform(intermediate)
        
        elif method == 'umap':
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=30,
                min_dist=0.1,
                random_state=random_state
            )
            return reducer.fit_transform(intermediate)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def identify_clusters(self, embedding, method='hdbscan', min_cluster_size=5):
        """
        Identify clusters in the reduced-dimensional embedding.
        
        Args:
            embedding: Low-dimensional embedding of accounts
            method: Clustering method ('hdbscan', 'dbscan', or 'kmeans')
            min_cluster_size: Minimum size of clusters
            
        Returns:
            Array of cluster labels
        """
        print(f"Identifying clusters with {method}...")
        
        if method == 'hdbscan':
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=3,
                cluster_selection_epsilon=0.1
            )
            return clusterer.fit_predict(embedding)
        
        elif method == 'dbscan':
            # Automatically determine epsilon based on data
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min_cluster_size)
            nn.fit(embedding)
            distances, _ = nn.kneighbors(embedding)
            
            # Use the average distance to the kth neighbor as epsilon
            epsilon = np.mean(distances[:, -1])
            
            clusterer = DBSCAN(eps=epsilon, min_samples=min_cluster_size)
            return clusterer.fit_predict(embedding)
        
        elif method == 'kmeans':
            # Determine optimal number of clusters using silhouette score
            from sklearn.metrics import silhouette_score
            
            # Try different numbers of clusters
            max_clusters = min(20, embedding.shape[0] // 5)
            best_score = -1
            best_k = 2
            
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(embedding)
                
                # Calculate silhouette score
                score = silhouette_score(embedding, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            print(f"Using {best_k} clusters based on silhouette score: {best_score:.4f}")
            kmeans = KMeans(n_clusters=best_k, random_state=42)
            return kmeans.fit_predict(embedding)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def create_static_plot(self, embedding, account_data, clusters=None, output_file=None):
        """
        Create a static visualization using matplotlib.
        
        Args:
            embedding: 2D embedding of accounts
            account_data: DataFrame containing account metadata
            clusters: Optional array of cluster labels
            output_file: Path to save the plot (if None, plot is displayed)
        """
        plt.figure(figsize=(12, 10))
        
        # Determine color scheme
        if clusters is not None:
            # Color by cluster
            scatter = plt.scatter(
                embedding[:, 0], embedding[:, 1],
                c=clusters, cmap='tab20',
                s=100 * np.log1p(account_data['followers']) / np.log1p(account_data['followers'].max()),
                alpha=0.7
            )
            
            # Add a legend for clusters
            unique_clusters = np.unique(clusters)
            handles = []
            labels = []
            
            for cluster in unique_clusters:
                if cluster == -1:
                    label = 'Noise'
                else:
                    label = f'Cluster {cluster}'
                
                handle = plt.Line2D(
                    [0], [0], marker='o', color='w',
                    markerfacecolor=scatter.cmap(scatter.norm(cluster)),
                    markersize=10
                )
                
                handles.append(handle)
                labels.append(label)
            
            plt.legend(handles, labels, loc='upper right')
        else:
            # Color by generation
            generations = account_data['generation'].fillna(-1).astype(int)
            scatter = plt.scatter(
                embedding[:, 0], embedding[:, 1],
                c=generations, cmap='viridis',
                s=100 * np.log1p(account_data['followers']) / np.log1p(account_data['followers'].max()),
                alpha=0.7
            )
            
            # Add a legend for generations
            unique_gens = np.unique(generations)
            handles = []
            labels = []
            
            gen_names = {
                0: 'Pre-2010',
                1: '2010-2012',
                2: '2013-2015',
                3: '2016-2018',
                4: '2019-2021',
                5: '2022+',
                -1: 'Unknown'
            }
            
            for gen in unique_gens:
                handle = plt.Line2D(
                    [0], [0], marker='o', color='w',
                    markerfacecolor=scatter.cmap(scatter.norm(gen)),
                    markersize=10
                )
                
                handles.append(handle)
                labels.append(gen_names.get(gen, f'Generation {gen}'))
            
            plt.legend(handles, labels, loc='upper right')
        
        # Annotate some points with artist names
        # Choose top accounts by followers
        top_idx = account_data['followers'].nlargest(15).index
        
        for idx in top_idx:
            plt.annotate(
                account_data.loc[idx, 'names'],
                (embedding[idx, 0], embedding[idx, 1]),
                fontsize=9,
                alpha=0.8,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.title('SoundCloud Community Landscape')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(linestyle='--', alpha=0.7)
        
        # Save or show the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        else:
            plt.show()
    
    def create_interactive_plot(self, embedding, account_data, clusters=None, output_file=None):
        """
        Create an interactive visualization using Plotly.
        
        Args:
            embedding: 2D embedding of accounts
            account_data: DataFrame containing account metadata
            clusters: Optional array of cluster labels
            output_file: Path to save the HTML plot
        """
        # Prepare data
        plot_data = account_data.copy()
        plot_data['x'] = embedding[:, 0]
        plot_data['y'] = embedding[:, 1]
        
        # Create hover text
        plot_data['hover_text'] = plot_data.apply(
            lambda row: f"<b>{row['names']}</b><br>" +
                       f"Followers: {row['followers']:,}<br>" +
                       f"Emerged: {row['emerge_date']}",
            axis=1
        )
        
        # Add cluster information if available
        if clusters is not None:
            plot_data['cluster'] = clusters
            color_column = 'cluster'
            color_discrete_map = {-1: '#808080'}  # Grey for noise points
        else:
            color_column = 'generation'
        
        # Determine point size based on followers (log scale)
        max_followers = plot_data['followers'].max()
        plot_data['point_size'] = 10 + 20 * (np.log1p(plot_data['followers']) / np.log1p(max_followers))
        
        # Create the plot
        if color_column == 'generation':
            # Create a custom color map for generations
            gen_names = {
                0: 'Pre-2010',
                1: '2010-2012',
                2: '2013-2015',
                3: '2016-2018',
                4: '2019-2021',
                5: '2022+',
                -1: 'Unknown'
            }
            
            plot_data['generation_name'] = plot_data['generation'].map(lambda x: gen_names.get(x, f'Generation {x}'))
            
            fig = px.scatter(
                plot_data,
                x='x', y='y',
                color='generation_name',
                size='point_size',
                hover_name='names',
                hover_data={
                    'x': False,
                    'y': False,
                    'point_size': False,
                    'generation_name': True,
                    'followers': True,
                    'emerge_date': True
                },
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        else:
            fig = px.scatter(
                plot_data,
                x='x', y='y',
                color='cluster',
                size='point_size',
                hover_name='names',
                hover_data={
                    'x': False,
                    'y': False,
                    'point_size': False,
                    'cluster': True,
                    'followers': True,
                    'emerge_date': True
                },
                color_discrete_sequence=px.colors.qualitative.Bold,
                color_discrete_map=color_discrete_map if 'cluster' in plot_data.columns else None,
                category_orders={'cluster': sorted(plot_data['cluster'].unique())}
            )
        
        # Update layout
        fig.update_layout(
            title='SoundCloud Community Landscape',
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            legend_title=color_column.capitalize(),
            template='plotly_white',
            hovermode='closest',
            width=1000,
            height=800
        )
        
        # Add links to SoundCloud profiles
        plot_data['url'] = 'https://soundcloud.com/' + plot_data['links']
        
        # Update hover template to include links
        fig.update_traces(
            hovertemplate='<b>%{hovertext}</b><br><br>' +
                          'Followers: %{customdata[3]:,}<br>' +
                          'Emerged: %{customdata[4]}<br>' +
                          '<extra></extra>'
        )
        
        # Save or show the plot
        if output_file:
            fig.write_html(output_file)
            print(f"Interactive plot saved to {output_file}")
        else:
            fig.show()
    
    def visualize(self, 
                  dim_reduction='umap', 
                  clustering='hdbscan',
                  min_interactions=10, 
                  max_dominance=0.4,
                  interactive=True,
                  show_clusters=True):
        """
        Process data and create visualizations.
        
        Args:
            dim_reduction: Dimensionality reduction method ('svd', 'tsne', or 'umap')
            clustering: Clustering method ('hdbscan', 'dbscan', or 'kmeans')
            min_interactions: Minimum interactions for an account to be included
            max_dominance: Maximum percentage of interactions that can go to a single account
            interactive: Whether to create an interactive plot
            show_clusters: Whether to show cluster assignments
        """
        # Preprocess data
        matrix, accounts = self.preprocess_data(
            min_interactions=min_interactions,
            max_dominance=max_dominance
        )
        
        # Reduce dimensions
        embedding = self.reduce_dimensions(matrix, method=dim_reduction)
        
        # Identify clusters if requested
        clusters = None
        if show_clusters:
            clusters = self.identify_clusters(embedding, method=clustering)
            
            # Add cluster information to accounts DataFrame
            accounts['cluster'] = clusters
            
            # Print cluster statistics
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            print("\nCluster statistics:")
            for cluster, count in cluster_counts.items():
                if cluster == -1:
                    print(f"  Noise points: {count}")
                else:
                    print(f"  Cluster {cluster}: {count} accounts")
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output filenames
        static_output = os.path.join(self.results_dir, f"soundcloud_landscape_{timestamp}.png")
        interactive_output = os.path.join(self.results_dir, f"soundcloud_landscape_interactive_{timestamp}.html")
        
        # Create plots
        if interactive:
            self.create_interactive_plot(
                embedding, 
                accounts, 
                clusters=clusters if show_clusters else None,
                output_file=interactive_output
            )
        
        # Always create a static plot
        self.create_static_plot(
            embedding, 
            accounts, 
            clusters=clusters if show_clusters else None,
            output_file=static_output
        )
        
        # Save processed data
        accounts.to_csv(os.path.join(self.results_dir, f"processed_accounts_{timestamp}.csv"), index=False)
        
        return embedding, accounts, clusters

if __name__ == "__main__":
    # Example usage
    visualizer = SoundCloudVisualizer(data_dir="./data")
    
    # Create visualizations with default settings
    embedding, accounts, clusters = visualizer.visualize(
        dim_reduction='umap',
        clustering='hdbscan',
        min_interactions=10,
        interactive=True,
        show_clusters=True
    ) 