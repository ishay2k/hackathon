import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier

preprocessing_data_path = 'data/preprocessed_data.csv' #TODO update with actual path hili+ishay
label_path = 'train_test_splits/train.labels.0.csv'

# Load the preprocessed data and labels
X = pd.read_csv(preprocessing_data_path)
Y = pd.read_csv(label_path)



