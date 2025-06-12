import pandas as pd


class Preprocess:

    def __init__(self, filepath):
        self.__data = pd.read_csv(filepath, encoding="utf-8")
        print(self.__data.columns)

        print(self.__data["אבחנה-T -Tumor mark (TNM)"].unique())

    def get_data(self):
        return self.__data

    def clean_age(self):
        """
        cleans the age column
        Returns
        -------
        """
        age_col = self.__data["אבחנה-Age"]
        valid_ages = age_col[(age_col >= 0) & (age_col <= 120)]
        mean_age = valid_ages.mean()

        self.__data.loc[(age_col < 0) | (age_col > 120), "אבחנה-Age"] = mean_age

    def clean_T_column(self):
        """
        Simplifies unique 'T' values in the TNM staging system by mapping subcategories
        (like 'T2a') to their main stage ('T2').
        """
        t_col = self.__data["T"]
        unique_values = t_col.unique()

        # Create a mapping dictionary
        value_map = {}
        for value in unique_values:
            if pd.isna(value) or value == "Not yet Established":
                value_map[value] = value  # Leave unchanged or handle later
            elif isinstance(value, str) and value.startswith("T") and len(value) > 2:
                for i in range(2, len(value)):
                    if not value[i].isdigit():
                        value_map[value] = value[:i]  # Simplify 'T2a' to 'T2'
                        break
                else:
                    value_map[value] = value  # Keep as-is if no letter found
            else:
                value_map[value] = value  # Keep normal values as-is

        # Replace values in the column using the map
        self.__data["T"] = t_col.map(value_map)


if __name__ == '__main__':
    data = Preprocess(r"C:\Users\ishay\IML\hackathon\train_test_splits\train.feats.csv")
