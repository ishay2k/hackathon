from sklearn.tree import DecisionTreeClassifier
import Preprocess
import pandas as pd
from sklearn.model_selection import train_test_split


class DecisionTreeClassifierWrapper:
    def __init__(self, max_depth=None, random_state=42):
        """
        Initializes the decision tree classifier wrapper.

        Parameters:
        -----------
        max_depth : int or None
            Maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
        random_state : int
            Controls the randomness of the estimator.
        """
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the decision tree model to the training data.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Labels (0 or 1).
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        """
        Predicts labels for the given data using the trained model.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix for prediction.

        Returns:
        --------
        np.ndarray
            Predicted labels (0 or 1).
        """
        return self.model.predict(X)


class baseLine1:
    def __init__(self, max_depth=5, random_state=42):
        """
        Initializes the decision tree classifier wrapper.

        Parameters:
        -----------
        max_depth : int or None
            Maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
        random_state : int
            Controls the randomness of the estimator.
        """
        self.models = [DecisionTreeClassifierWrapper(max_depth, random_state) for _ in range(11)]

        self.__preprocess = Preprocess.Preprocess(r"C:\Users\hilib\PycharmProjects\IML\hackathon\hackathon\train_test_splits\train.feats.csv",
                      r"C:\Users\hilib\PycharmProjects\IML\hackathon\hackathon\train_test_splits\train.labels.0.csv",
                      r"C:\Users\hilib\PycharmProjects\IML\hackathon\hackathon\train_test_splits\train.labels.1.csv"
                      )

        self.train = self.__preprocess.encode_dataframe()
        self.lable = self.__preprocess.encode_lable_0()

        # פיצול ראשוני ל-60% שמורות בצד ו-40% לפיצול נוסף
        X_temp, X_side, y_temp, y_side = train_test_split(self.train, self.lable, test_size=0.6, random_state=42)

        # פיצול ה-40% הנותרים ל-20% train ו-20% test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


    def mark_rows_with_value(self, target_value: int) -> pd.DataFrame:
        """
        Returns a DataFrame of the same size, with 1 if the list in the row contains the target_value, else 0.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with a single column, where each cell contains a list of integers.
        target_value : int
            The value to search for inside each list.

        Returns:
        --------
        pd.DataFrame
            A DataFrame with a single column of 0s and 1s, same index as the input.
        """
        column_name = self.y_train.columns[0]
        binary_series = self.y_train[column_name].apply(lambda lst: 1 if target_value in lst else 0)
        return binary_series.to_frame(name=f'contains_{target_value}')

    def fit(self):
        print(self.__preprocess.get_metastases())


if __name__ == '__main__':
    b = baseLine1()
    b.fit()