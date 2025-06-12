from itertools import product

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from lightgbm import record_evaluation
from lightgbm import LGBMRegressor, record_evaluation

from hackathon.src.Preprocess import Preprocess

preprocessor = Preprocess(
        r"../train_test_splits/train.feats.csv",
        r"../train_test_splits/train.labels.0.csv",
        r"../train_test_splits/train.labels.1.csv"
    )
#best hyperParameter
n_estimators = 200
num_leaves = 128


def load_and_prepare_data():
    """
    Loads features and multilabel targets from CSV files.
    Adds a synthetic random feature to serve as a baseline for feature importance.
    Shuffles the dataset to ensure randomness.
    Returns the feature matrix and label matrix.
    """


    all_train = preprocessor.encode_dataframe()
    all_lables = preprocessor.get_labels_1()

    return all_train, all_lables



# === Model training and evaluation ===
def train_and_evaluate_regression_model(X_train, X_val, Y_train, Y_val, params,record_curve = False):
    """
    Trains a filtered regression model with given parameters and evaluates on the validation set.
    If record_curve is True, returns the evals_result_ to plot training/validation loss curves.
    """
    model = LGBMRegressor(**params, random_state=42)
    model.fit(X_train, Y_train)

    evals_result = {}
    model.fit(
        X_train,
        Y_train,
        eval_set=[(X_train, Y_train), (X_val, Y_val)],
        eval_metric='l2',
        callbacks=[record_evaluation(evals_result)],
    )

    # Predict on validation set
    val_preds = model.predict(X_val)

    # Evaluate model using Mean Squared Error (lower is better)
    mse = mean_squared_error(Y_val, val_preds)
    print(f"Validation MSE: {mse:.4f}")
    return model, mse, evals_result if record_curve else None


# === Predict and save ===
def predict_and_save_test_results(model, X_test_filtered):
    """
    Makes predictions on the test set and saves them to a CSV file.
    """
    predictions = model.predict(X_test_filtered)

    # Wrap predictions in a DataFrame with required column name
    predictions_df = pd.DataFrame(predictions, columns=["tumor_size_mm"])
    predictions_df.to_csv("regression_predictions.csv", index=False)
    print("Saved predictions to regression_predictions.csv")


# === Hyperparameter tuning ===
def grid_search_regression(X_train_filtered, X_val_filtered, Y_train, Y_val):
    """
    Performs grid search on combinations of n_estimators and num_leaves.
    Records results and saves them to a CSV file.
    """
    n_estimators = [100, 150, 200]
    num_leaves = [31, 64, 128]

    best_score = float('inf')
    best_model = None
    best_params = None
    results = []

    for i, (n_est, num_leaf) in enumerate(product(n_estimators, num_leaves)):
        params = {'n_estimators': n_est, 'num_leaves': num_leaf}
        print(f"\n===== Grid Config {i+1}: {params} =====")
        model, mse, _ = train_and_evaluate_regression_model(X_train_filtered, X_val_filtered, Y_train, Y_val, params)
        results.append({"config": f"Config {i+1}", **params, "mse": mse})

        if mse < best_score:
            best_score = mse
            best_model = model
            best_params = params

    print(f"\n🏆 Best configuration: {best_params} with MSE = {best_score:.4f}")

    pd.DataFrame(results).to_csv("grid_search_results.csv", index=False)
    print("Saved grid search results to grid_search_results.csv")
    return best_model

# === Diagnostic comparison of model types ===
def compare_model_behaviors(X_train, X_val, Y_train, Y_val):
    """
    Compares underfit, baseline, and overfit models by plotting their training/validation loss curves.
    """
    configs = {
        'Underfit': {'n_estimators': 10, 'num_leaves': 8, 'max_depth': 3},
        'Baseline': {'n_estimators': 100, 'num_leaves': 31},
        'Overfit': {'n_estimators': 1000, 'num_leaves': 512, 'min_child_samples': 1}
    }

    plt.figure(figsize=(8, 5))
    for name, params in configs.items():
        print(f"\n🔍 Training {name} model...")
        _, _, evals_result = train_and_evaluate_regression_model(
            X_train, X_val, Y_train, Y_val, params, record_curve=True
        )
        val_loss = evals_result['valid_1']['l2']
        plt.plot(val_loss, label=f"{name}")

    plt.title("Validation MSE vs Boosting Iterations")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Validation MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# === Main pipeline ===
def main():
    """
    Main execution function to run all parts of the regression modeling and evaluation process.
    """

    # Load features and raw labels
    X, Y_raw = load_and_prepare_data()  # Y_raw is a DataFrame with list-like entries

    # X_temp, X_side, y_temp, y_side = train_test_split(X,
    #                                                   Y_raw,
    #                                                   test_size=0.6,
    #                                                   random_state=42)
    #
    # # פיצול ה-40% הנותרים ל-20% train ו-20% test
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y_raw,
                                                              test_size=0.2,
                                                              random_state=42)

    # Align test data to same feature space
    compare_model_behaviors(X_train, X_val, Y_train, Y_val)

    # Train and tune model - best hyperParameters chosen
    # best_model = grid_search_regression(X_train, X_val, Y_train, Y_val)

    params = {'n_estimators': n_estimators, 'num_leaves': num_leaves}
    # best_model, mse, _ = train_and_evaluate_regression_model(X_train, X_val, Y_train, Y_val, params)

    # Predict and save results
    test_preprocessor = Preprocess(
        r"../train_test_splits/test.feats.csv",
        r"../train_test_splits/train.labels.0.csv",
        r"../train_test_splits/train.labels.1.csv"
    )

    best_model = LGBMRegressor(**params, random_state=42)
    best_model.fit(X,Y_raw)

    X_test = test_preprocessor.encode_dataframe()
    predict_and_save_test_results(best_model, X_test)


if __name__ == "__main__":
    main()
