import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os


class Unsupervised:
    def kmeans_cluster_and_plot(df: pd.DataFrame, column_name: str, n_clusters: int, output_path: str):
        """
        Applies KMeans clustering to a single column of a DataFrame and saves a plot of the clustering result.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data.
        column_name : str
            The name of the column to cluster.
        n_clusters : int
            The number of clusters to use in KMeans.
        output_path : str
            The full path (including filename) where the plot will be saved.
        """
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")

        # Drop NA and reshape
        data = df[[column_name]].dropna()

        # Ensure it's numeric
        data = pd.to_numeric(data[column_name], errors='coerce').dropna().to_frame()

        # KMeans fitting
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        data['cluster'] = kmeans.fit_predict(data[[column_name]])

        # Plot
        plt.figure(figsize=(10, 6))
        for cluster_id in range(n_clusters):
            cluster_data = data[data['cluster'] == cluster_id]
            plt.scatter(cluster_data.index, cluster_data[column_name], label=f'Cluster {cluster_id}')

        plt.title(f'KMeans Clustering on {column_name}')
        plt.xlabel('Index')
        plt.ylabel(column_name)
        plt.legend()
        plt.tight_layout()

        # Save the plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

