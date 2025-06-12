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
    return pred_y, converted_y

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

def tune_baseline_hyperparameters(X_train, X_val, Y_train, Y_val, n_iter=10, cv=2):
    """
    Performs Randomized Search over baseline LightGBM hyperparameters using macro F1 as the metric.
    Trains the best model and evaluates it on the validation set.
    Parameters:
        - X_train, Y_train: training data
        - X_val, Y_val: validation data
        - n_iter: number of random parameter combinations to try
        - cv: number of cross-validation folds
    Returns:
        - best_model: the trained model with best hyperparameters
        - best_params: the dictionary of best parameters
    """
    print(f"\n🎯 Starting baseline hyperparameter tuning (RandomizedSearchCV, {n_iter} iterations, {cv}-fold)...")

    param_dist = {
        'estimator__n_estimators': [100, 200, 300, 400],
        'estimator__num_leaves': [31, 63, 127],
        'estimator__learning_rate': [0.1, 0.05, 0.03],
        'estimator__max_depth': [5, 7, 9, -1]
    }

    macro_scorer = make_scorer(f1_score, average='macro', zero_division=0)

    base_model = OneVsRestClassifier(LGBMClassifier(random_state=42, verbose=-1))

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        scoring=macro_scorer,
        cv=cv,
        n_iter=n_iter,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, Y_train)

    print("✅ Best parameters found:", search.best_params_)

    # Evaluate best model on validation set
    best_model = search.best_estimator_
    preds_val, _ = _predict(best_model, X_val)
    macro_val = f1_score(Y_val, preds_val, average='macro', zero_division=0)
    micro_val = f1_score(Y_val, preds_val, average='micro', zero_division=0)

    print(f"🎯 Tuned Baseline - Macro F1 on Val: {macro_val:.4f}")
    print(f"🎯 Tuned Baseline - Micro F1 on Val: {micro_val:.4f}")

    return best_model, search.best_params_

from sklearn.preprocessing import MultiLabelBinarizer

# === Predict and save ===
def predict_and_save_test_results(model, X_test_filtered, filename="predictions_part1.csv"):
    """
    Makes predictions on the test set using a multi-label classification model
    and saves them to a CSV file.

    Parameters:
    -----------
    model : sklearn classifier
        Trained OneVsRestClassifier model.
    X_test_filtered : pd.DataFrame
        Preprocessed feature matrix for the test set.
    filename : str
        Name of the output CSV file.
    """
    # Perform prediction and convert to readable format
    predictions, label_lists = _predict(model, X_test_filtered)

    # Save readable label list format (e.g., ['label1', 'label2'])
    label_lists.to_csv(filename, index=False)
    print(f"✅ Saved predictions to {filename}")



def compare_model_variants(X_train, X_val, Y_train, Y_val, tuned_baseline_params=None):
    """
    Trains and compares underfit, baseline, and overfit model variants using F1 scores.
    Allows overriding the baseline model with tuned hyperparameters.
    """
    configs = {
        'Underfit': {'n_estimators': 10, 'num_leaves': 8, 'max_depth': 3},
        'Baseline': tuned_baseline_params or {'n_estimators': 300, 'num_leaves': 31, 'max_depth': 7, 'learning_rate': 0.05},
        'Overfit': {'n_estimators': 1000, 'num_leaves': 512, 'min_child_samples': 1}
    }

    results_macro = {}
    results_micro = {}
    models = []
    for name, params in configs.items():
        print(f"\n🔍 Training {name} model with params: {params}")
        model = OneVsRestClassifier(
            LGBMClassifier(**params, random_state=42, verbose=-1)
        )
        #save the model
        models.append(model)
        model.fit(X_train, Y_train)

        preds_train, _ = _predict(model, X_train)
        preds_val, _ = _predict(model, X_val)

        preds_train = pd.DataFrame(preds_train)
        preds_val = pd.DataFrame(preds_val)

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
        return models[1]  # Return the baseline model for further use

def main():
    """
    Loads the data, splits it, performs feature selection and compares model variants.
    Uses MultiLabelBinarizer to handle multi-label classification correctly.
    """

    # Load features and raw labels
    X, Y_raw = load_and_prepare_data()  # Y_raw is a DataFrame with list-like entries

    # שלב 1: פיצול ל־60% train, 20% hyperparameter tuning, 20% test
    X_train, X_temp, Y_train_raw, Y_temp_raw = train_test_split(X, Y_raw, test_size=0.4, random_state=42)
    X_hyper, X_test, Y_hyper_raw, Y_test_raw = train_test_split(X_temp, Y_temp_raw, test_size=0.5, random_state=42)

    # Apply MultiLabelBinarizer to handle lists like [1, 2]
    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(Y_train_raw.iloc[:, 0])
    Y_hyper = mlb.transform(Y_hyper_raw.iloc[:, 0])
    Y_test = mlb.transform(Y_test_raw.iloc[:, 0])

    # Feature selection using label 0's model
    # important_features = get_important_features(X_train, Y_train)

    # Filter by important features
    X_train_filtered = X_train
    X_hyper_filtered = X_hyper
    X_test_filtered = X_test

     ####### TODO - dont delete this comment ########
    # שלב 2: טיונינג על קבוצת hyper בלבד
    # tuned_model, best_params = tune_baseline_hyperparameters(X_train_filtered, X_hyper_filtered, Y_train, Y_hyper)
    #
    # print(f"🎯 Best tuned model parameters: {best_params}")

    # שלב 3: השוואת מודלים על קבוצת test לפי ההיפר־פרמטרים של הבייסליין
    # compare_model_variants(
    #     X_train_filtered, X_test_filtered, Y_train, Y_test,
    #     tuned_baseline_params={key.replace("estimator__", ""): val for key, val in best_params.items()}
    # )
    best_model = compare_model_variants(
        X_train_filtered, X_test_filtered, Y_train, Y_test
    )

    params = {'n_estimators': 300, 'num_leaves': 31, 'max_depth': 7, 'learning_rate': 0.05}
    test_preprocessor = Preprocess(
        r"../train_test_splits/test.feats.csv",
        r"../train_test_splits/train.labels.0.csv",
        r"../train_test_splits/train.labels.1.csv"
    )

    model = OneVsRestClassifier(
        LGBMClassifier(**params, random_state=42, verbose=-1)
    )
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(Y_raw.iloc[:, 0])
    model.fit(X, Y)

    X_test = test_preprocessor.encode_dataframe()
    predict_and_save_test_results(best_model, X_test)


if __name__ == "__main__":
    main()
