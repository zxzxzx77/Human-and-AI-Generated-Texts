import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import random
import tensorflow as tf

# Load the dataset
dataset = pd.read_csv("/content/DatasetPart1-20-100Words.csv")
dataset.dropna(inplace=True)

# Separate features (text) and target variable
X = dataset['Text']
y = dataset['Class']

# Convert class labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.1, random_state=42)

# Define function to evaluate features using ABC algorithm
def evaluate_features_ABC(solution, X_train, y_train):
    selected_features = [i for i, val in enumerate(solution) if val == 1]
    sorted_values = X_train[:, selected_features].toarray()  
    num_samples = X_train.shape[0]
    num_features = len(selected_features)  
    X_reshaped = tf.reshape(sorted_values, [num_samples, num_features])
    
    mlp_input = Input(shape=(num_features,))
    mlp_dense = Dense(128, activation='relu')(mlp_input)
    mlp_output = Dense(1, activation='sigmoid')(mlp_dense)
    
    model = Model(inputs=mlp_input, outputs=mlp_output)
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
    history = model.fit(X_reshaped, y_train, epochs=10, batch_size=64, verbose=0)
    accuracy = model.evaluate(X_reshaped, y_train, verbose=0)[1]
    
    print("Accuracy for each epoch:", history.history['accuracy'])
    
    return accuracy,

# Define function to create MLP architecture with dynamic input shape
def create_mlp_model(num_features):
    mlp_input = Input(shape=(num_features,))
    mlp_dense = Dense(128, activation='relu')(mlp_input)
    mlp_output = Dense(1, activation='sigmoid')(mlp_dense)
    model = Model(inputs=mlp_input, outputs=mlp_output)
    return model

# Define ABC parameters
num_employed_bees = 5
num_onlooker_bees = 5
num_iterations = 2

# Initialize best_solution_ABC before the loop
best_solution_ABC = None

# Initialize ABC algorithm
def initialize_population(num_features):
    population = []
    for _ in range(num_employed_bees + num_onlooker_bees):
        solution = [random.randint(0, 1) for _ in range(num_features)]
        population.append(solution)
    return population

# Run the ABC algorithm
population = initialize_population(X_tfidf.shape[1])
for iteration in range(num_iterations):
    print("Iteration:", iteration + 1)
    for solution in population:
        fitness = evaluate_features_ABC(solution, X_train, y_train)
        if best_solution_ABC is None or fitness > evaluate_features_ABC(best_solution_ABC, X_train, y_train):
            best_solution_ABC = solution

# Get the number of selected features from the best solution found by the ABC algorithm
num_selected_features = sum(best_solution_ABC)

# Create the MLP model with dynamic input shape based on the number of selected features
model_mlp = create_mlp_model(num_selected_features)

# Compile the model
model_mlp.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Train and evaluate the MLP model with the selected features
X_reshaped_ABC_train = X_train[:, best_solution_ABC].toarray()
X_reshaped_ABC_test = X_test[:, best_solution_ABC].toarray()

# Get the number of selected features from the best solution found by the ABC algorithm
num_selected_features = sum(best_solution_ABC)

# Create the MLP model with dynamic input shape based on the number of selected features
model_mlp = create_mlp_model(X_reshaped_ABC_train.shape[1])

# Compile the model
model_mlp.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Train and evaluate the MLP model with the selected features
model_mlp.fit(X_reshaped_ABC_train, y_train, epochs=10, batch_size=64, verbose=0)
loss_ABC, accuracy_ABC = model_mlp.evaluate(X_reshaped_ABC_test, y_test, verbose=0)
print("Accuracy on testing data (MLP with ABC-selected features):", accuracy_ABC)



