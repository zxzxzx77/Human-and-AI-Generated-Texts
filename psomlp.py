import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from pyswarm import pso

# Load dataset (example for Dataset 1)
df = pd.read_csv('/content/DatasetPart1-20-100Words.csv')  # Replace with actual dataset path
X = df['Text'].values
y = df['Class'].values

# Data preprocessing
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(X).toarray()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define MLP model
def create_mlp(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define evaluation function for PSO
def evaluate(individual):
    selected_features = [i for i, bit in enumerate(individual) if bit >= 0.5]
    if len(selected_features) == 0:
        return 1  # PSO minimizes the objective function, return high error when no features are selected

    X_train_sel = X_train[:, selected_features]
    X_test_sel = X_test[:, selected_features]

    model = create_mlp(input_dim=len(selected_features))
    model.fit(X_train_sel, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred = (model.predict(X_test_sel) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    return 1 - accuracy  # PSO minimizes the objective function

# PSO parameters
lb = [0] * X_train.shape[1]  # Lower bounds
ub = [1] * X_train.shape[1]  # Upper bounds
swarm_size = 10
max_iter = 5

# Run PSO
best_individual, best_cost = pso(evaluate, lb, ub, swarmsize=swarm_size, maxiter=max_iter)

selected_features = [i for i, bit in enumerate(best_individual) if bit >= 0.5]

# Train final MLP model with selected features
X_train_sel = X_train[:, selected_features]
X_test_sel = X_test[:, selected_features]
final_model = create_mlp(input_dim=len(selected_features))
final_model.fit(X_train_sel, y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate final model
y_pred = (final_model.predict(X_test_sel) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
