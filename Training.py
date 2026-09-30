from pathlib import Path

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
BASE_DIR = Path(__file__).resolve().parent
data = pd.read_csv(BASE_DIR / "Combined Data.csv")

# Drop unnecessary columns (if present)
data = data.drop(columns=['Unnamed: 0'], errors='ignore')

# Convert 'status' to binary labels: Normal -> 0, Others -> 1
data['status'] = (data['status'] != 'Normal').astype(int)

# Handle missing values
data = data.dropna(subset=['statement'])

# Split features and target
X = data['statement']
y = data['status']

# Split before fitting the vectorizer to avoid leaking test-set vocabulary.
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train, y_train)

# Predict on test data
y_pred = lr_model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and vectorizer
with open(BASE_DIR / "model.pkl", "wb") as model_file:
    pickle.dump(lr_model, model_file)

with open(BASE_DIR / "vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")
