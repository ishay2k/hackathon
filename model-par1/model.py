import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

print("✅ LightGBM is working!")
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from src.Preprocess import Preprocess

preprocessor = Preprocess(
        r"../train_test_splits/train.feats.csv",
        r"../train_test_splits/train.labels.0.csv",
        r"../train_test_splits/train.labels.1.csv"
    )

def load_and_prepare_data():
    """
    Loads features and multilabel targets from CSV files.
    Adds a synthetic random feature to serve as a baseline for feature importance.
    Shuffles the dataset to ensure randomness.
    Returns the feature matrix and label matrix.
    """


    all_train = preprocessor.encode_dataframe()
    all_lables = preprocessor.encode_lable_0()

    return all_train, all_lables

# def get_important_features(X_train, Y_train):
#     """
#     Trains a baseline OneVsRest LightGBM model and computes feature importances.
#     Retains features whose importance is greater than the random feature.
#     Plots feature importances and returns selected features.
#     """
#     model = OneVsRestClassifier(LGBMClassifier(n_estimators=100, num_leaves=255, random_state=42))
#     model.fit(X_train, Y_train)
#
#     importances = model.estimators_[0].feature_importances_
#     importance_series = pd.Series(importances, index=X_train.columns)
#
#     # Plot feature importances
#     importance_series.sort_values().plot(kind='barh', figsize=(8,12))
#     plt.title('Feature Importances for Label 0')
#     plt.tight_layout()
#     plt.show()
#
#     # Keep only features more important than random
#     threshold = importance_series["Random_Feature"]
#     important_features = importance_series[importance_series > threshold].index.tolist()
#     print(f"Using {len(important_features)} important features")
#
#     return important_features


def top_k_predictions(y_proba, k=3, threshold=0.2):
    """
    Returns binary predictions with at most k labels per instance,
    only if their probability is >= threshold.
    """
    y_pred = np.zeros_like(y_proba, dtype=int)
    for i, probs in enumerate(y_proba):
        top_k_idx = probs.argsort()[::-1][:k]  # indices of top-k highest
        for j in top_k_idx:
            if probs[j] >= threshold:
                y_pred[i, j] = 1
    return y_pred

def convert_binary_df_to_label_list(df):
    """
    Converts a binary DataFrame (multi-hot encoded) into a single-column DataFrame
    where each cell contains a list of label indices with value 1.

    Example:
    input:
        0 1 2 3
        1 0 1 0
        0 0 1 1

    output:
        labels
        [0, 2]
        [2, 3]
    """
    return pd.DataFrame({
        'labels': df.apply(lambda row: list(row[row == 1].index.astype(int)), axis=1)
    })

def map_values_in_list_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps each numeric value in a list column to a corresponding string label using a predefined dictionary.
    If the mapped list contains only '__EMPTY__', it is replaced with an empty list ([]).

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame with a single column where each cell is a list of numeric label indices.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the same structure, where each numeric list has been mapped to string labels,
        and lists containing only '__EMPTY__' are replaced with [].
    """
    mapping_dict = preprocessor.get_metastases()
    inverse_dict = {v: k for k, v in mapping_dict.items()}  # Reverse the mapping

    column_name = df.columns[0]
    df_copy = df.copy()

    def map_and_clean(lst):
        mapped = [inverse_dict.get(val, val) for val in lst]
        if mapped == ["__EMPTY__"]:
            return []
        return mapped

    df_copy[column_name] = df_copy[column_name].apply(map_and_clean)
    return df_copy


def _predict(model, X):
    pred_y = model.predict(X)
    pred_y = top_k_predictions(pred_y, k=3, threshold=0.2)
    converted_y = convert_binary_df_to_label_list(pd.DataFrame(pred_y))
    converted_y = map_values_in_list_column(converted_y)
    # print(pd.DataFrame(converted_y).head(50))
    return pred_y, converted_y


def compare_model_variants(X_train, X_val, Y_train, Y_val):
    """
    Trains and compares underfit, baseline, and overfit model variants using F1 scores.
    Suppresses LightGBM warnings and avoids sklearn UndefinedMetricWarnings.
    """
    configs = {
        'Underfit': {'n_estimators': 10, 'num_leaves': 8, 'max_depth': 3},
        'Baseline': {'n_estimators': 100, 'num_leaves': 31},
        'Overfit': {'n_estimators': 1000, 'num_leaves': 512, 'min_child_samples': 1}
    }

    results_macro = {}
    results_micro = {}

    for name, params in configs.items():
        print(f"\n🔍 Training {name} model with params: {params}")
        model = OneVsRestClassifier(
            LGBMClassifier(**params, random_state=42, verbose=-1)
        )
        model.fit(X_train, Y_train)


        # preds_train = model.predict(X_train)
        # preds_train = top_k_predictions(preds_train, k=3, threshold=0.2)
        preds_train, converted_preds_train = _predict(model, X_train)
        # print(np.sum(preds_train, axis=1).max())

        preds_train = pd.DataFrame(preds_train)
        # preds_val = model.predict(X_val)
        # preds_val = top_k_predictions(preds_val, k=3, threshold=0.2)
        preds_val, converted_preds_val = _predict(model, X_val)

        # print(f"val = {np.sum(preds_val, axis=1).max()}")


        macro_train = f1_score(Y_train, preds_train, average='macro', zero_division=0)
        macro_val = f1_score(Y_val, preds_val, average='macro', zero_division=0)
        micro_train = f1_score(Y_train, preds_train, average='micro', zero_division=0)
        micro_val = f1_score(Y_val, preds_val, average='micro', zero_division=0)

        results_macro[name] = (macro_train, macro_val)
        results_micro[name] = (micro_train, micro_val)

        print(f"Macro F1 - Train: {macro_train:.4f}, Val: {macro_val:.4f}")
        print(f"Micro F1 - Train: {micro_train:.4f}, Val: {micro_val:.4f}")

    for metric_name, results in zip(["Macro F1", "Micro F1"], [results_macro, results_micro]):
        labels = list(results.keys())
        train_scores = [results[label][0] for label in labels]
        val_scores = [results[label][1] for label in labels]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width / 2, train_scores, width, label='Train')
        ax.bar(x + width / 2, val_scores, width, label='Validation')

        ax.set_ylabel('F1 Score')
        ax.set_title(f'Model Comparison: {metric_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

from sklearn.preprocessing import MultiLabelBinarizer

def main():
    """
    Loads the data, splits it, performs feature selection and compares model variants.
    Uses MultiLabelBinarizer to handle multi-label classification correctly.
    """


    # Load features and raw labels
    X, Y_raw = load_and_prepare_data()  # Y_raw is a DataFrame with list-like entries


    X_temp, X_side, y_temp, y_side = train_test_split(X,
                                                      Y_raw,
                                                      test_size=0.6,
                                                      random_state=42)

    # פיצול ה-40% הנותרים ל-20% train ו-20% test
    X_train, X_val, Y_train_raw, Y_val_raw = train_test_split(X_temp, y_temp,
                                                        test_size=0.5,
                                                        random_state=42)


    # Apply MultiLabelBinarizer to handle lists like [1, 2]
    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(Y_train_raw.iloc[:, 0])
    Y_val = mlb.transform(Y_val_raw.iloc[:, 0])

    # Feature selection using label 0's model
    # important_features = get_important_features(X_train, Y_train)

    # Filter by important features
    X_train_filtered = X_train
    X_val_filtered = X_val

    # if len(important_features) == 0:
    #     print("⚠️ No important features found. Using all features.")
    #     X_train_filtered = X_train
    #     X_val_filtered = X_val
    # else:
    #     X_train_filtered = X_train[important_features]
    #     X_val_filtered = X_val[important_features]

    # Compare model variants
    compare_model_variants(X_train_filtered, X_val_filtered, Y_train, Y_val)


if __name__ == "__main__":
    main()
