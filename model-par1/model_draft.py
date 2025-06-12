import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from src.Preprocess import Preprocess
import re

def clean_feature_names(df):
    """
    Remove special or duplicate characters from column names
    and make them LightGBM-safe.
    """
    def sanitize(col):
        col = re.sub(r'[^A-Za-z0-9_]', '_', col)  # Replace anything not alphanumeric or underscore
        col = re.sub(r'_+', '_', col)  # Replace multiple underscores with one
        col = col.strip('_')  # Strip leading/trailing underscores
        return col

    def deduplicate_columns(columns):
        seen = {}
        new_columns = []
        for col in columns:
            if col not in seen:
                seen[col] = 0
                new_columns.append(col)
            else:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
        return new_columns

    df.columns = [sanitize(col) for col in df.columns]
    df.columns = deduplicate_columns(df.columns)
    return df

def load_and_prepare_data():
    """
    Loads preprocessed features and multilabel targets using a Preprocess class.
    Adds a synthetic random feature to serve as a baseline for feature importance.
    Shuffles the dataset to ensure randomness.
    Returns the feature matrix and label matrix.
    """
    preprocessor = Preprocess(
        r"../train_test_splits/train.feats.csv",
        r"../train_test_splits/train.labels.0.csv",
        r"../train_test_splits/train.labels.1.csv"
    )
    X = preprocessor.create_dummies()
    Y = preprocessor.encode_lable_0()

    X["Random_Feature"] = np.random.rand(len(X))
    X, Y = shuffle(X, Y, random_state=42)

    X = clean_feature_names(X)

    return X, Y

def split_data(X, Y):
    """
    Splits the dataset into training (20%), validation (20%), and remaining (60%) sets.
    Returns these sets and the chunk size (5% of the full dataset).
    """
    train_ratio = 0.2
    val_ratio = 0.2
    chunk_size = int(0.05 * len(X))

    X_train_val, X_remaining, Y_train_val, Y_remaining = train_test_split(X, Y, test_size=1 - (train_ratio + val_ratio), random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

    return X_train, Y_train, X_val, Y_val, X_remaining, Y_remaining, chunk_size

def get_important_features(X_train, Y_train):
    """
    Trains a baseline OneVsRest LightGBM model and computes feature importances.
    Retains features whose importance is greater than the random feature.
    Plots feature importances and returns selected features.
    """
    model = OneVsRestClassifier(LGBMClassifier(n_estimators=100, num_leaves=255, random_state=42))
    model.fit(X_train, Y_train)

    importances = model.estimators_[0].feature_importances_
    importance_series = pd.Series(importances, index=X_train.columns)

    importance_series.sort_values().plot(kind='barh', figsize=(8,12))
    plt.title('Feature Importances for Label 0')
    plt.tight_layout()
    plt.show()

    threshold = importance_series["Random_Feature"]
    important_features = importance_series[importance_series > threshold].index.tolist()
    print(f"Using {len(important_features)} important features")

    return important_features

def compare_model_variants(X_train, X_val, Y_train, Y_val):
    """
    Trains and compares underfit, baseline, and overfit model variants using macro and micro F1 scores.
    Plots F1 scores for train and validation sets to illustrate performance differences.
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
        model = OneVsRestClassifier(LGBMClassifier(**params, random_state=42))
        model.fit(X_train, Y_train)

        preds_train = model.predict(X_train)
        preds_val = model.predict(X_val)

        macro_train = f1_score(Y_train, preds_train, average='macro')
        macro_val = f1_score(Y_val, preds_val, average='macro')
        micro_train = f1_score(Y_train, preds_train, average='micro')
        micro_val = f1_score(Y_val, preds_val, average='micro')

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

def main():
    """
    Main pipeline to run data preparation, feature selection,
    and variant comparison without touching the test set.
    """
    X, Y = load_and_prepare_data()
    X_train, Y_train, X_val, Y_val, X_remaining, Y_remaining, chunk_size = split_data(X, Y)

    important_features = get_important_features(X_train, Y_train)
    X_train_filtered = X_train[important_features]
    X_val_filtered = X_val[important_features]

    compare_model_variants(X_train_filtered, X_val_filtered, Y_train, Y_val)

if __name__ == "__main__":
    main()
