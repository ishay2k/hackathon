from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import Preprocess
from sklearn.model_selection import train_test_split

class BaseLint2:
    def __init__(self, random_state=42):
        """
        Initializes the linear regression model.
        """
        self.model = LinearRegression()
        self.__preprocess = Preprocess.Preprocess(r"C:\Users\hilib\PycharmProjects\IML\hackathon\hackathon\train_test_splits\train.feats.csv",
                      r"C:\Users\hilib\PycharmProjects\IML\hackathon\hackathon\train_test_splits\train.labels.0.csv",
                      r"C:\Users\hilib\PycharmProjects\IML\hackathon\hackathon\train_test_splits\train.labels.1.csv"
                      )

        self.train = self.__preprocess.encode_dataframe()
        self.lable = self.__preprocess.get_labels_1()

        # פיצול ראשוני ל-60% שמורות בצד ו-40% לפיצול נוסף
        X_temp, X_side, y_temp, y_side = train_test_split(self.train, self.lable, test_size=0.6, random_state=42)

        # פיצול ה-40% הנותרים ל-20% train ו-20% test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


    def fit(self):
        """
        Fits the linear regression model to the training data.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target values.
        """
        X = self.X_train
        y = self.y_train
        self.model.fit(X, y)

    def predict(self) -> pd.DataFrame:
        """
        Predicts values for the given input features.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix for prediction.

        Returns:
        --------
        np.ndarray
            Predicted target values.
        """
        X = self.X_test
        return pd.DataFrame(self.model.predict(X))

if __name__ == '__main__':
    b = BaseLint2()
    b.fit()
    print(b.predict().head(50))

