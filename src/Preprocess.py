import pandas as pd


class Preprocess:

    def __init__(self, filepath):
        self.__data = pd.read_csv(filepath, encoding="utf-8")
        print(self.__data.columns)

        print(self.__data["surgery before or after-Actual activity"].head(20))

    def get_data(self):
        return self.__data


if __name__ == '__main__':
    data = Preprocess(r"C:\Users\ishay\IML\hackathon\train_test_splits\train.feats.csv")
