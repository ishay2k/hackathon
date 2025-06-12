import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from lightgbm import record_evaluation
from lightgbm import LGBMRegressor, record_evaluation




# === Load and prepare data ===
def load_and_prepare_regression_data():
    """
    Loads features and tumor size labels for regression.
    Adds a random feature to help filter unimportant features.
    Shuffles the data for unbiased training.
    """
    feats_path = 'data/preprocessed_data.csv'
    label_path = 'train_test_splits/train.labels.1.csv'
    test_path = 'train_test_splits/test.feats.csv'

    X = pd.read_csv(feats_path)
    Y = pd.read_csv(label_path)
    X_test = pd.read_csv(test_path)

    # Add a noise feature to use as feature importance threshold
    # This feature will help us define a cutoff for important features
    X['Random_Feature'] = np.random.rand(len(X))
    X_test['Random_Feature'] = np.random.rand(len(X_test))

    # Shuffle the data to ensure the model is trained on a representative sample
    X, Y = shuffle(X, Y, random_state=42)

    return X, Y.squeeze(), X_test


# === Train/validation split ===
def split_train_val(X, Y):
    """
    Splits the data into training and validation sets (80/20).
    """
    return train_test_split(X, Y, test_size=0.2, random_state=42)


# === Feature selection ===
def select_important_features(X_train, Y_train):
    """
    Trains a LGBMRegressor and selects features with importance above random noise.
    """
    model = LGBMRegressor(n_estimators=100, num_leaves=31, random_state=42)
    model.fit(X_train, Y_train)

    importances = model.feature_importances_  # array of importance scores
    importance_series = pd.Series(importances, index=X_train.columns)

    # Plot importances to visualize which features the model relies on
    importance_series.sort_values().plot(kind='barh', figsize=(8, 10))
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()

    # Use the random feature as a noise threshold
    threshold = importance_series["Random_Feature"]
    important_features = importance_series[importance_series > threshold].index.tolist()
    print(f"✅ {len(important_features)} features kept (out of {X_train.shape[1]})")

    return important_features


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
    Performs grid search on several LGBM hyperparameter configurations.
    Records results and saves them to a CSV file.
    """
    # Define a small set of parameter combinations to evaluate
    grid = [
        {'n_estimators': 100, 'num_leaves': 31},
        {'n_estimators': 200, 'num_leaves': 64},
        {'n_estimators': 150, 'num_leaves': 128},
    ]
    best_score = float('inf')
    best_model = None
    best_params = None
    results = []

    # Loop through each configuration
    for i, params in enumerate(grid):
        print(f"\n===== Grid Config {i+1} =====")
        model, mse = train_and_evaluate_regression_model(X_train_filtered, X_val_filtered, Y_train, Y_val, params)
        results.append({"config": f"Config {i+1}", **params, "mse": mse})

        # Save the best model based on validation MSE
        if mse < best_score:
            best_score = mse
            best_model = model
            best_params = params

    print(f"\n🏆 Best configuration: {best_params} with MSE = {best_score:.4f}")

    # Save all grid search results for comparison
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
    # Load, clean, and augment data
    X, Y, X_test = load_and_prepare_regression_data()

    # Split training/validation
    X_train, X_val, Y_train, Y_val = split_train_val(X, Y)

    # Feature selection
    important_features = select_important_features(X_train, Y_train)

    # Reduce to important features
    X_train_filtered = X_train[important_features]
    X_val_filtered = X_val[important_features]

    # Align test data to same feature space
    X_test_filtered = X_test.reindex(columns=important_features, fill_value=0)


    compare_model_behaviors(X_train_filtered, X_val_filtered, Y_train, Y_val)

    # Train and tune model
    best_model = grid_search_regression(X_train_filtered, X_val_filtered, Y_train, Y_val)

    # Predict and save results
    predict_and_save_test_results(best_model, X_test_filtered)


if __name__ == "__main__":
    main()
