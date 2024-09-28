import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load the dataset using a relative path
data_path = os.path.join(os.getcwd(), 'Data Source', 'SPAM.csv')
df = pd.read_csv(data_path)

# Preprocess the data
X = df['Message']
y = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)  # Convert labels to binary

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the model
model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)

# Save the model and feature vectorizer using relative paths
pickle_dir = os.path.join(os.getcwd(), 'Pickle Files')
os.makedirs(pickle_dir, exist_ok=True)  # Ensure the directory exists

model_path = os.path.join(pickle_dir, 'model.pkl')
feature_path = os.path.join(pickle_dir, 'feature.pkl')

with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(feature_path, 'wb') as feature_file:
    pickle.dump(tfidf, feature_file)

print("Model and feature vectorizer saved successfully.")
