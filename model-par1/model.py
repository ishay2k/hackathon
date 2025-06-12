import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

def load_and_prepare_data():
    """
    Loads the preprocessed features, labels, and test data.
    Adds a random feature for feature importance thresholding.
    Shuffles the data to ensure proper train/validation splits.
    """
    preprocessing_data_path = 'data/preprocessed_data.csv' #TODO update with actual path hili+ishay
    label_path = 'train_test_splits/train.labels.0.csv'
    test_data_path = 'train_test_splits/test.labels.0.csv' #TODO update with actual path hili+ishay

    X = pd.read_csv(preprocessing_data_path)
    Y = pd.read_csv(label_path)
    X_test = pd.read_csv(test_data_path)

    # Add a synthetic feature with random noise to use as importance baseline
    X["Random_Feature"] = np.random.rand(len(X))
    X_test["Random_Feature"] = np.random.rand(len(X_test))

    # Shuffle the training data to ensure randomness
    X, Y = shuffle(X, Y, random_state=42)

    return X, Y, X_test

def split_data(X, Y):
    """
    Splits the data into a fixed training set (20%) and a remaining set for evaluation (80%).
    Calculates the size of each evaluation chunk (5%).
    """
    train_size = int(0.2 * len(X))
    chunk_size = int(0.05 * len(X))

    X_train = X.iloc[:train_size].reset_index(drop=True)
    Y_train = Y.iloc[:train_size].reset_index(drop=True)
    X_remaining = X.iloc[train_size:].reset_index(drop=True)
    Y_remaining = Y.iloc[train_size:].reset_index(drop=True)

    return X_train, Y_train, X_remaining, Y_remaining, chunk_size

def get_important_features(X_train, Y_train):
    """
    Trains a baseline model and computes feature importances.
    Filters features whose importance exceeds that of a random feature.
    """
    model = OneVsRestClassifier(LGBMClassifier(n_estimators=100, num_leaves=255, random_state=42))
    model.fit(X_train, Y_train)

    # Extract feature importances from the first label's model
    importances = model.estimators_[0].feature_importances_
    importance_series = pd.Series(importances, index=X_train.columns)

    # Visualize feature importances
    importance_series.sort_values().plot(kind='barh', figsize=(8,12))
    plt.title('Feature Importances for Label 0')
    plt.tight_layout()
    plt.show()

    # Use the random feature's importance as a noise threshold
    threshold = importance_series["Random_Feature"]

    # Keep only features more important than the noise
    important_features = importance_series[importance_series > threshold].index.tolist()
    print(f"Using {len(important_features)} important features")

    return important_features

def train_and_save_baseline(X_train, Y_train, X_test, important_features, label_columns):
    """
    Trains a baseline model using only the selected important features.
    Saves predictions on the test set to a CSV file.
    """
    # Select only the filtered features from training and test sets
    X_train_filtered = X_train[important_features]
    X_test_filtered = X_test.reindex(columns=important_features, fill_value=0)

    model = OneVsRestClassifier(LGBMClassifier(n_estimators=100, num_leaves=255, random_state=42))
    model.fit(X_train_filtered, Y_train)

    # Predict and export test results
    test_preds = model.predict(X_test_filtered)
    pd.DataFrame(test_preds, columns=label_columns).to_csv("filtered_predictions_baseline.csv", index=False) #TODO open the file
    print("Saved baseline predictions to filtered_predictions_baseline.csv")

    return X_train_filtered, X_test_filtered

def evaluate_model_on_next_chunk(X_train_filtered, Y_train, X_remaining, Y_remaining, important_features, chunk_size, params, chunk_idx):
    """
    Evaluates the model with the given hyperparameters on a specific chunk of the remaining data.
    """
    # Define the chunk boundaries
    start = chunk_idx * chunk_size
    end = start + chunk_size

    # Slice the next chunk for evaluation
    X_val = X_remaining.iloc[start:end][important_features]
    Y_val = Y_remaining.iloc[start:end]

    # Train the model with custom hyperparameters
    model = OneVsRestClassifier(LGBMClassifier(**params, random_state=42))
    model.fit(X_train_filtered, Y_train)

    # Evaluate the model using F1 scores
    preds = model.predict(X_val)
    micro = f1_score(Y_val, preds, average="micro")
    macro = f1_score(Y_val, preds, average="macro")
    print("\n🔍 Evaluating with params:", params)
    print(f"Chunk {chunk_idx+1}: Micro F1 = {micro:.4f}, Macro F1 = {macro:.4f}")

def evaluate_model_on_fixed_chunks(X_train_filtered, Y_train, X_remaining, Y_remaining, important_features, chunk_size, params):
    """
    Evaluates the model using the same 5 fixed chunks (each 5%) of the remaining data.
    """
    results = []

    # Train the model once on the training data
    model = OneVsRestClassifier(LGBMClassifier(**params, random_state=42))
    model.fit(X_train_filtered, Y_train)
    print("\n📌 Static evaluation with params:", params)

    # Evaluate on 5 fixed validation chunks
    for i in range(5):
        start = i * chunk_size
        end = start + chunk_size
        X_val = X_remaining.iloc[start:end][important_features]
        Y_val = Y_remaining.iloc[start:end]
        preds = model.predict(X_val)
        micro = f1_score(Y_val, preds, average="micro")
        macro = f1_score(Y_val, preds, average="macro")
        print(f"Chunk {i+1}: Micro F1 = {micro:.4f}, Macro F1 = {macro:.4f}")
        results.append((micro, macro))
    return results

def grid_search_configurations(X_train_filtered, Y_train, X_remaining, Y_remaining, important_features, chunk_size):
    """
    Runs a grid search over a few parameter combinations and prints evaluation results.
    """
    grid = [
        {'n_estimators': 100, 'num_leaves': 64},
        {'n_estimators': 150, 'num_leaves': 128},
        {'n_estimators': 200, 'num_leaves': 100},
    ]
    for i, params in enumerate(grid):
        print(f"\n===== Config {i+1} =====")
        evaluate_model_on_fixed_chunks(X_train_filtered, Y_train, X_remaining, Y_remaining, important_features, chunk_size, params)

def main():
    """
    Main execution function to run all parts of the modeling and evaluation process.
    """
    X, Y, X_test = load_and_prepare_data()
    X_train, Y_train, X_remaining, Y_remaining, chunk_size = split_data(X, Y)
    important_features = get_important_features(X_train, Y_train)
    X_train_filtered, _ = train_and_save_baseline(X_train, Y_train, X_test, important_features, Y.columns)

    # Evaluate a sample config on the next validation chunk
    params_example = {'n_estimators': 120, 'num_leaves': 100}
    evaluate_model_on_next_chunk(X_train_filtered, Y_train, X_remaining, Y_remaining, important_features, chunk_size, params_example, chunk_idx=0)

    # Try multiple configurations using grid search
    grid_search_configurations(X_train_filtered, Y_train, X_remaining, Y_remaining, important_features, chunk_size)

if __name__ == "__main__":
    main()






