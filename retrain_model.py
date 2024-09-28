
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv(r'A:\sakhi2\Spam-Email-Detection-main\Data Source\SPAM.csv')

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

# Save the model and feature vectorizer
with open('A:\\sakhi2\\Spam-Email-Detection-main\\Pickle Files\\model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('A:\\sakhi2\\Spam-Email-Detection-main\\Pickle Files\\feature.pkl', 'wb') as feature_file:
    pickle.dump(tfidf, feature_file)

print("Model and feature vectorizer saved successfully.")
