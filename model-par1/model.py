import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    """
    Loads preprocessed features and multilabel targets from CSV files.
    Adds a synthetic random feature to serve as a baseline for feature importance.
    Shuffles the dataset to ensure randomness.
    Returns the feature matrix, label matrix, and test set.
    """
    preprocessing_data_path = 'data/preprocessed_data.csv'
    label_path = 'train_test_splits/train.labels.0.csv'
    test_data_path = 'train_test_splits/test.labels.0.csv'

    # Load feature and label datasets
    X = pd.read_csv(preprocessing_data_path)
    Y = pd.read_csv(label_path)
    X_test = pd.read_csv(test_data_path)

    # Add random noise feature to help identify unimportant features
    X["Random_Feature"] = np.random.rand(len(X))
    X_test["Random_Feature"] = np.random.rand(len(X_test))

    # Shuffle data to avoid order bias
    X, Y = shuffle(X, Y, random_state=42)

    return X, Y, X_test

def split_data(X, Y):
    """
    Splits the dataset into a fixed training set (20%) and a remaining set (80%) for evaluation.
    Calculates the chunk size as 5% of the full dataset.
    Returns train, remaining data, and chunk size.
    """
    train_size = int(0.2 * len(X))
    chunk_size = int(0.05 * len(X))

    # Split training and remaining datasets
    X_train = X.iloc[:train_size].reset_index(drop=True)
    Y_train = Y.iloc[:train_size].reset_index(drop=True)
    X_remaining = X.iloc[train_size:].reset_index(drop=True)
    Y_remaining = Y.iloc[train_size:].reset_index(drop=True)

    return X_train, Y_train, X_remaining, Y_remaining, chunk_size

def get_important_features(X_train, Y_train):
    """
    Trains a baseline OneVsRest LightGBM model and computes feature importances.
    Retains features whose importance is greater than the random feature.
    Plots feature importances and returns selected features.
    """
    # Train LightGBM model on multilabel classification task
    model = OneVsRestClassifier(LGBMClassifier(n_estimators=100, num_leaves=255, random_state=42))
    model.fit(X_train, Y_train)

    # Extract feature importances from first label's model (assumption)
    importances = model.estimators_[0].feature_importances_
    importance_series = pd.Series(importances, index=X_train.columns)

    # Plot the importance scores
    importance_series.sort_values().plot(kind='barh', figsize=(8,12))
    plt.title('Feature Importances for Label 0')
    plt.tight_layout()
    plt.show()

    # Keep only features that are more important than the random one
    threshold = importance_series["Random_Feature"]
    important_features = importance_series[importance_series > threshold].index.tolist()
    print(f"Using {len(important_features)} important features")

    return important_features

def train_and_save_baseline(X_train, Y_train, X_test, important_features, label_columns):
    """
    Trains a baseline OneVsRest LightGBM model on selected features.
    Saves predictions on the test set to a CSV file.
    Returns the filtered training and test sets.
    """
    # Select only the important features
    X_train_filtered = X_train[important_features]
    X_test_filtered = X_test.reindex(columns=important_features, fill_value=0)

    # Train model and make predictions
    model = OneVsRestClassifier(LGBMClassifier(n_estimators=100, num_leaves=255, random_state=42))
    model.fit(X_train_filtered, Y_train)

    test_preds = model.predict(X_test_filtered)

    # Save predictions to CSV
    pd.DataFrame(test_preds, columns=label_columns).to_csv("filtered_predictions_baseline.csv", index=False)
    print("Saved baseline predictions to filtered_predictions_baseline.csv")

    return X_train_filtered, X_test_filtered

def compare_model_variants(X_train, X_val, Y_train, Y_val):
    """
    Trains and compares underfit, baseline, and overfit model variants using macro and micro F1 scores.
    Plots F1 scores for train and validation sets to illustrate performance differences.
    """
    # Define model configurations for comparison
    configs = {
        'Underfit': {'n_estimators': 10, 'num_leaves': 8, 'max_depth': 3},
        'Baseline': {'n_estimators': 100, 'num_leaves': 31},
        'Overfit': {'n_estimators': 1000, 'num_leaves': 512, 'min_child_samples': 1}
    }

    results_macro = {}
    results_micro = {}

    # Train each model and compute F1 scores
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

    # Plot comparison results
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
    baseline model training, and variant comparison.
    """
    # Load and prepare data
    X, Y, X_test = load_and_prepare_data()

    # Split into training and evaluation sets
    X_train, Y_train, X_remaining, Y_remaining, chunk_size = split_data(X, Y)

    # Select important features based on feature importance
    important_features = get_important_features(X_train, Y_train)

    # Train baseline model and save predictions
    X_train_filtered, _ = train_and_save_baseline(X_train, Y_train, X_test, important_features, Y.columns)

    # Split training data again for validation purposes
    X_subtrain, X_val, Y_subtrain, Y_val = train_test_split(X_train_filtered, Y_train, test_size=0.2, random_state=42)

    # Compare model variants on validation set
    compare_model_variants(X_subtrain, X_val, Y_subtrain, Y_val)

if __name__ == "__main__":
    main()
