import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier

def load_data():
    """
    Loads full training data and test data (for final submission only).
    Adds a synthetic random feature for consistency with feature selection.
    """
    X_train = pd.read_csv("train.feats.csv")
    Y_train = pd.read_csv("train.labels.0.csv")
    X_test = pd.read_csv("test.feats.csv")

    # Add same random feature
    X_train["Random_Feature"] = np.random.rand(len(X_train))
    X_test["Random_Feature"] = np.random.rand(len(X_test))

    return X_train, Y_train, X_test

def get_important_features(X, Y):
    """
    Select important features based on LightGBM feature importance vs random baseline.
    """
    model = OneVsRestClassifier(LGBMClassifier(n_estimators=100, num_leaves=255, random_state=42))
    model.fit(X, Y)

    importances = model.estimators_[0].feature_importances_
    importance_series = pd.Series(importances, index=X.columns)

    threshold = importance_series["Random_Feature"]
    important_features = importance_series[importance_series > threshold].index.tolist()
    print(f"Using {len(important_features)} important features for final model")

    return important_features

def train_and_predict(X, Y, X_test, features, label_columns):
    """
    Trains model on full data and predicts on test set.
    Saves predictions to CSV.
    """
    X_train_filtered = X[features]
    X_test_filtered = X_test.reindex(columns=features, fill_value=0)

    model = OneVsRestClassifier(LGBMClassifier(n_estimators=100, num_leaves=255, random_state=42))
    model.fit(X_train_filtered, Y)

    predictions = model.predict(X_test_filtered)
    pd.DataFrame(predictions, columns=label_columns).to_csv("predictions.csv", index=False)
    print("✅ Predictions saved to predictions.csv")

def main():
    X, Y, X_test = load_data()
    important_features = get_important_features(X, Y)
    train_and_predict(X, Y, X_test, important_features, Y.columns)

if __name__ == "__main__":
    main()
