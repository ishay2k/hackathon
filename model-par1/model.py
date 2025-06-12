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


def load_and_prepare_data():
    """
    Loads features and multilabel targets from CSV files.
    Adds a synthetic random feature to serve as a baseline for feature importance.
    Shuffles the dataset to ensure randomness.
    Returns the feature matrix and label matrix.
    """

    preprocessor = Preprocess(
        r"../train_test_splits/train.feats.csv",
        r"../train_test_splits/train.labels.0.csv",
        r"../train_test_splits/train.labels.1.csv"
    )

    X = preprocessor.encode_dataframe()
    Y = preprocessor.encode_lable_0()

    # feats_path = 'train.feats.csv'
    # labels_path = 'train.labels.0.csv'
    #
    # # Load features and labels
    # X = pd.read_csv(feats_path)
    # Y = pd.read_csv(labels_path)

    # Add random noise feature
    X["Random_Feature"] = np.random.rand(len(X))

    # Shuffle data to avoid order bias
    X, Y = shuffle(X, Y, random_state=42)

    return X, Y

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

    # Plot feature importances
    importance_series.sort_values().plot(kind='barh', figsize=(8,12))
    plt.title('Feature Importances for Label 0')
    plt.tight_layout()
    plt.show()

    # Keep only features more important than random
    threshold = importance_series["Random_Feature"]
    important_features = importance_series[importance_series > threshold].index.tolist()
    print(f"Using {len(important_features)} important features")

    return important_features

def compare_model_variants(X_train, X_val, Y_train, Y_val):
    """
    Trains and compares underfit, baseline, and overfit model variants using F1 scores.
    Plots F1 scores for train and validation sets.
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

from sklearn.preprocessing import MultiLabelBinarizer

def main():
    """
    Loads the data, splits it, performs feature selection and compares model variants.
    Uses MultiLabelBinarizer to handle multi-label classification correctly.
    """
    # Load features and raw labels
    X, Y_raw = load_and_prepare_data()  # Y_raw is a DataFrame with list-like entries

    # Split BEFORE encoding so alignment with X is preserved
    X_train, X_val, Y_train_raw, Y_val_raw = train_test_split(
        X, Y_raw, test_size=0.2, random_state=42
    )

    # Apply MultiLabelBinarizer to handle lists like [1, 2]
    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(Y_train_raw.iloc[:, 0])
    Y_val = mlb.transform(Y_val_raw.iloc[:, 0])

    # Feature selection using label 0's model
    important_features = get_important_features(X_train, Y_train)

    # # Filter by important features
    # X_train_filtered = X_train[important_features]
    # X_val_filtered = X_val[important_features]

    if len(important_features) == 0:
        print("⚠️ No important features found. Using all features.")
        X_train_filtered = X_train
        X_val_filtered = X_val
    else:
        X_train_filtered = X_train[important_features]
        X_val_filtered = X_val[important_features]

    # Compare model variants
    compare_model_variants(X_train_filtered, X_val_filtered, Y_train, Y_val)


if __name__ == "__main__":
    main()
