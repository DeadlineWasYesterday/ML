import os

import pandas as pd
from sklearn.svm import SVR


# Get the data folder path relative to this file
data_folder = os.path.join(os.path.dirname(__file__), "data")

# Get the paths to each dataset
human_power_dataset_1 = os.path.join(data_folder, "humanpower1.csv")
human_power_dataset_2 = os.path.join(data_folder, "humanpower2.csv")

df1 = pd.read_csv(human_power_dataset_1)
df2 = pd.read_csv(human_power_dataset_2)

# Concat both datasets
df = pd.concat([df1, df2])

# Get the dataframe colums as ndarrays
X = df["o2"].values.reshape(-1, 1)  # Reshape the X values to a 2D array
y = df["wattsPerKg"].values

regressor = SVR()

# Fit the model according to the training data
regressor.fit(X, y)

# Predict a human's power output for a given O2 measurement
prediction = regressor.predict([[35]])
confidence = regressor.score(X, y)

print(prediction, confidence)

