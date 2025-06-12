from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import Preprocess
class PCA_Class:
    def __init__(self):
        self.__data = Preprocess.Preprocess(r"C:\Users\ishay\IML\hackathon\train_test_splits\train.feats.csv",
                      r"C:\Users\ishay\IML\hackathon\train_test_splits\train.labels.0.csv",
                      r"C:\Users\ishay\IML\hackathon\train_test_splits\train.labels.1.csv"
                      )

        self.__data = self.__data.encode_dataframe()

    def perform_PCA(self):
        numeric_df = self.__data.select_dtypes(include='number').dropna()

        # Step 3: Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # Step 4: Apply PCA
        pca = PCA(n_components=2)  # Change to desired number of components
        pca_result = pca.fit_transform(scaled_data)

        # Step 5: Visualize the results
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA - First 2 Principal Components")
        plt.grid(True)
        plt.show()
        plt.close()

if __name__ == '__main__':
    pca = PCA_Class()
    pca.perform_PCA()