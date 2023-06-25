import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# Load the data from the Excel file
data = pd.read_excel("E:\\Users\\HP\\Desktop\\Anshuman's Folder\\YUVAi\\Database\\SD\\TrainingData.xlsx")

# Extract the grades and difficulty levels
grades = data['Physics'].values.reshape(-1, 1)
difficulty_levels = data['Difficulty level'].values

# Train a logistic regression model
model = LogisticRegression()
model.fit(grades, difficulty_levels)

# Save the trained model to a file
joblib.dump(model, 'trained_model.joblib')
