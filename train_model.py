import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Download required nltk data
for resource in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Transform text
df['transformed'] = df['message'].apply(transform_text)

# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(df['transformed']).toarray()
y = pd.get_dummies(df['label'], drop_first=True).values.ravel()

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))

print("✅ Model and vectorizer trained and saved successfully!")