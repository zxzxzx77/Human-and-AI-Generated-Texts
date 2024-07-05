import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
dataset = pd.read_csv("/content/DatasetPart1-20-100Words.csv")

# Drop rows with missing values
dataset.dropna(inplace=True)

# Separate features (text) and target variable
X = dataset['Text']
y = dataset['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize the Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=500, max_depth=8, max_features='sqrt', random_state=42)

# Train the classifier on the training data
random_forest_classifier.fit(X_train_tfidf, y_train)

# Predict the target variable on the testing data
y_pred = random_forest_classifier.predict(X_test_tfidf)

# Calculate the accuracy of the classifier on the testing data
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on testing data:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')

# Calculate recall
recall = recall_score(y_test, y_pred, average='weighted')

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')

# Print precision, recall, and F1 score
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
