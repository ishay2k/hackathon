import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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
# Add the random feature to test data
X_test["Random_Feature"] = np.random.rand(len(X_test))

X, Y = shuffle(X, Y, random_state=42)  # Shuffle the data to ensure randomness
train_size = int(0.2 * len(X))  # 20% for training
chunk_size = int(0.05 * len(X))  # Split into 5 chunks

X_train = X.iloc[:train_size]
Y_train = Y.iloc[:train_size]

X_remaining = X.iloc[train_size:]
Y_remaining = Y.iloc[train_size:]

val_chunks = []
for i in range(5):
    start = i * chunk_size
    end = start + chunk_size
    X_chunk = X_remaining.iloc[start:end]
    Y_chunk = Y_remaining.iloc[start:end]
    val_chunks.append((X_chunk, Y_chunk))

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


#get list of important features
important_features = importance_series[importance_series > threshold].index.tolist()

# filter the original data to keep only important features
X_filtered = X[important_features]
# Add the random feature to test data
X_test["Random_Feature"] = np.random.rand(len(X_test))
# Ensure test data has the same columns
X_test_filtered = X_test.reindex(columns=important_features, fill_value=0)

# split the data into train and test sets
X_train, X_val, Y_train, Y_val = train_test_split(X_filtered, Y, test_size=0.2, random_state=42)

# Initialize the OneVsRestClassifier with LGBMClassifier
model_filtered = OneVsRestClassifier(LGBMClassifier(n_estimators=100, num_leaves=255, random_state=42))
# Fit the model with filtered features
model_filtered.fit(X_train, Y_train)

# Produce predictions on the validation set
val_predictions = model_filtered.predict(X_val)
# Check the predictions quality on the validation set
print("Micro F1 (Filtered Features):", f1_score(Y_val, val_predictions, average="micro"))
print("Macro F1 (Filtered Features):", f1_score(Y_val, val_predictions, average="macro"))

test_preds = model_filtered.predict(X_test_filtered)
# Save the filtered predictions to a CSV file
filtered_predictions_df = pd.DataFrame(test_preds, columns=Y.columns)
filtered_predictions_df.to_csv('filtered_predictions.csv', index=False)  #TODO open the file
print("Filtered predictions saved to 'filtered_predictions.csv'")









