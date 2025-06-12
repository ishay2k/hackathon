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
        # self.models = [DecisionTreeClassifierWrapper(max_depth, random_state) for _ in range(11)]
        self.models = {}
        self.max_depth = max_depth
        self.random_state = random_state

        self.__preprocess = Preprocess.Preprocess(r"C:\Users\hilib\PycharmProjects\IML\hackathon\hackathon\train_test_splits\train.feats.csv",
                      r"C:\Users\hilib\PycharmProjects\IML\hackathon\hackathon\train_test_splits\train.labels.0.csv",
                      r"C:\Users\hilib\PycharmProjects\IML\hackathon\hackathon\train_test_splits\train.labels.1.csv"
                      )

        self.train = self.__preprocess.encode_dataframe()
        self.lable = self.__preprocess.encode_lable_0()


    def get_train(self):
        return self.train

    def get_lable(self):
        return self.lable

    def get__preprocess(self):
        return self.__preprocess

    def mark_rows_with_value(self, y, target_value: int) -> pd.DataFrame:
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
        column_name = y.columns[0]
        binary_series = y[column_name].apply(lambda lst: 1 if target_value in lst else 0)
        return binary_series.to_frame(name=f'contains_{target_value}')

    def fit(self, X, y):
        dict = self.__preprocess.get_metastases()

        for key, val in dict.items():
            lables = self.mark_rows_with_value(y, val)
            self.models[val] = DecisionTreeClassifierWrapper(self.max_depth, self.random_state)
            self.models[val].fit(X, lables)

    def map_values_in_list_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ממירה כל מספר במערך לפי המילון שניתן.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame עם עמודה אחת שמכילה מערכים של מספרים.

        Returns:
        --------
        pd.DataFrame
            DataFrame עם אותם אינדקסים, כשהערכים הומרו לפי המילון.
        """
        mapping_dict = self.__preprocess.get_metastases()

        column_name = df.columns[0]
        df_copy = df.copy()

        df_copy[column_name] = df_copy[column_name].apply(
            lambda lst: [mapping_dict.get(val, val) for val in lst]
        )
        return df_copy

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts labels for the given data using all trained models.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix for prediction.

        Returns:
        --------
        pd.DataFrame
            A DataFrame with a single column, where each cell contains a list
            of metastasis codes (ints) predicted to be present (value=1).
        """
        if not self.models:
            raise ValueError("No trained models found. Run fit() before predict().")

        metastases_dict = self.__preprocess.get_metastases()  # {'LYM - Lymph nodes': 1, ...}
        inverse_dict = {v: k for k, v in metastases_dict.items()}  # רק אם תרצי גם להפוך בעתיד

        predictions = {}

        for name, val in metastases_dict.items():
            if val in self.models:
                preds = self.models[val].predict(X)
                predictions[val] = preds  # שמור לפי המספר, לא לפי השם

        # הפוך ל-DataFrame
        pred_df = pd.DataFrame(predictions, index=X.index)

        # המר כל שורה לרשימת המספרים שחזוי עבורם 1
        result_series = pred_df.apply(lambda row: [int(col) for col in row.index if row[col] == 1], axis=1)

        # החזר DataFrame עם עמודה אחת
        return pd.DataFrame({'predicted_metastases': result_series})

    def loss(self, X):
        pred_y = self.predict(X)



if __name__ == '__main__':
    b = baseLine1()
    # פיצול ראשוני ל-60% שמורות בצד ו-40% לפיצול נוסף
    X_temp, X_side, y_temp, y_side = train_test_split(b.get_train(), b.get_lable(), test_size=0.6, random_state=42)

    # פיצול ה-40% הנותרים ל-20% train ו-20% test
    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                                            random_state=42)

    b.fit(X_train, y_train)
    print(b.predict(X_test).head(20))
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%")

    print(b.map_values_in_list_column(y_test.head(20)))
