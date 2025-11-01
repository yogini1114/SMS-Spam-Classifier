import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load your dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

# Convert labels (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorization
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_cv, y_train)

# Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)

print("✅ Model and vectorizer saved successfully!")