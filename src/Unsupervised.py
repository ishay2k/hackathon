import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import Preprocess

class Unsupervised:
    def __init__(self):
        self.__data = Preprocess.Preprocess(r"C:\Users\ishay\IML\hackathon\train_test_splits\train.feats.csv",
                      r"C:\Users\ishay\IML\hackathon\train_test_splits\train.labels.0.csv",
                      r"C:\Users\ishay\IML\hackathon\train_test_splits\train.labels.1.csv"
                      )

        self.__data = self.__data.encode_dataframe()

    def kmeans_cluster_and_plot(self, df: pd.DataFrame, n_clusters: int, output_path: str):
        """
        Applies KMeans clustering to all numeric columns in the DataFrame
        and saves a 2D PCA plot of the clustering result.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data.
        n_clusters : int
            The number of clusters to use in KMeans.
        output_path : str
            The full path (including filename) where the plot will be saved.
        """

        #Select numeric columns only
        numeric_df = df.select_dtypes(include=['number']).dropna()

        if numeric_df.empty:
            raise ValueError("No numeric data available after dropping NA.")

        #Normalize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        # KMeans fitting
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(scaled_data)

        # Reduce to 2 dimensions using PCA for visualization
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)

        # Plot
        plt.figure(figsize=(10, 6))
        for cluster_id in range(n_clusters):
            plt.scatter(
                reduced_data[clusters == cluster_id, 0],
                reduced_data[clusters == cluster_id, 1],
                label=f'Cluster {cluster_id}'
            )

        plt.title('KMeans Clustering (PCA projection)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.tight_layout()

        # Save the plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.show()
        plt.close()



    def cluster(self):
        # print(self.__data.get_data().unique())
        self.kmeans_cluster_and_plot(self.__data[["Form Name", "Stage", "Her2"]], 4,
                                     r"C:\Users\ishay\IML\hackathon")


if __name__ == '__main__':
    u = Unsupervised()
    u.cluster()