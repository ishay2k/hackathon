import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

preprocessing_data_path = 'data/preprocessed_data.csv' #TODO update with actual path hili+ishay
label_path = 'train_test_splits/train.labels.0.csv'
test_data_path = 'train_test_splits/test.labels.0.csv' #TODO update with actual path hili+ishay

# Load the preprocessed data and labels
X = pd.read_csv(preprocessing_data_path)
Y = pd.read_csv(label_path)
# Load the test data
X_test = pd.read_csv(test_data_path)

# Add a random feature for demonstration
X["Random_Feature"] = np.random.rand(len(X))

# Initialize the OneVsRestClassifier with LGBMClassifier
model = OneVsRestClassifier(LGBMClassifier(n_estimators=100,num_leaves=255, random_state=42))

# Fit the model
model.fit(X, Y)

#feture importances - find the feature importances of the first estimator
importances = model.estimators_[0].feature_importances_
# Create a pandas Series for feature importances
importance_series = pd.Series(importances, index=X.columns)
# Sort the feature importances and plot them
importance_series = importance_series.sort_values().plot(kind='barh', figsize=(8,12))
plt.title('Feature Importances for Label 0')
plt.tight_layout()
plt.show()

# Get the threshold for the random feature
threshold = importance_series["Random_Feature"]
print(f"Features less important than the random feature: \n")
print(importance_series[importance_series < threshold])

# produce predictions
X_test["Random_Feature"] = np.random.rand(len(X_test))  # Add the random feature to test data
X_test = X_test.reindex(columns=X.columns, fill_value=0)  # Ensure test data has the same columns
predictions = model.predict(X_test)
# Save the predictions to a CSV file
predictions_df = pd.DataFrame(predictions, columns=Y.columns)
predictions_df.to_csv('predictions.csv', index=False) #TODO open the file

# check the predictions quality
train_predictions = model.predict(X)
print("Micro F1:", f1_score(Y, train_predictions, average="micro"))
print("Macro F1:", f1_score(Y, train_predictions, average="macro"))








