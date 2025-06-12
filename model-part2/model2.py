import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#  load the preprocessed data
preprocessing_data_path = 'data/preprocessed_data.csv'  # TODO update with actual path
test_data_path = 'train_test_splits/test.feats.csv'  # or correct path to test features

X = pd.read_csv(preprocessing_data_path)
Y = pd.read_csv('train_test_splits/train.labels.1.csv')
# Load the test data
X_test = pd.read_csv(test_data_path)  # TODO update with actual path

# Add a random feature for demonstration
X["Random_Feature"] = np.random.rand(len(X))
X_test["Random_Feature"] = np.random.rand(len(X_test))  # Add the random feature to test data

# ensure test data has the same columns
X_test = X_test.reindex(columns=X.columns, fill_value=0)

#split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

#train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

#calculate feature importances
importances = model.feature_importances_
# Create a pandas Series for feature importances
importance_series = pd.Series(importances, index=X.columns)

#plot feature importances
importance_series.sort_values().plot(kind='barh', figsize=(8, 10))
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

# Get the threshold for the random feature
threshold = importance_series["Random_Feature"]
important_features = importance_series[importance_series > threshold].index.tolist()
print(f"✅ {len(important_features)} features kept (out of {X.shape[1]})")

#retrain the model with only important features
X_train_filtered = X_train[important_features]
X_val_filtered = X_val[important_features]
X_test_filtered = X_test[important_features]

model_filtered = RandomForestRegressor(n_estimators=100, random_state=42)
model_filtered.fit(X_train_filtered, Y_train)


# produce predictions
val_preds = model_filtered.predict(X_val_filtered)
mse = mean_squared_error(Y_val, val_preds)
print(f"Validation MSE: {mse}")

# Save the predictions to a CSV file
predictions = model_filtered.predict(X_test_filtered)
predictions_df = pd.DataFrame(predictions, columns=["tumor_size_mm"]) #TODO check the column name
predictions_df.to_csv('regression_predictions.csv', index=False)  # TODO open the file
