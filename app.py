# app.py
import streamlit as st
import pandas as pd
import string
import nltk
from nltk.data import find  # used to check presence of resources

# --- Ensure NLTK data is present before importing corpora/tokenizers ---
def ensure_nltk_resources():
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
    }
    for pkg, path in resources.items():
        try:
            find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

ensure_nltk_resources()

# Now safe to import these
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

# --- text transform function (uses cached stopwords & stemmer) ---
def transform_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = word_tokenize(text)  # requires punkt
    words = [t for t in tokens if t.isalnum()]  # alphanumeric tokens
    filtered = [ps.stem(w) for w in words if w not in STOPWORDS and w not in string.punctuation]
    return " ".join(filtered)

# --- load dataset function (safe) ---
@st.cache_data
def load_dataset(path: str = "spam.csv"):
    df = pd.read_csv(path, encoding='latin-1', low_memory=False)
    # keep only required columns if they exist
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    else:
        # try fallback names
        df = df.rename(columns=lambda c: c.strip().lower())
        # ensure 'label' and 'message' exist
        if 'label' not in df.columns or 'message' not in df.columns:
            raise RuntimeError("CSV doesn't contain expected columns 'v1','v2' or 'label','message'.")
        df = df[['label', 'message']]
    return df

# --- training function cached as resource (runs once) ---
@st.cache_resource
def train_model_from_df(df: pd.DataFrame):
    # Transform messages (this is cached)
    df = df.copy()
    df['transformed'] = df['message'].apply(transform_text)

    cv = CountVectorizer()
    X = cv.fit_transform(df['transformed']).toarray()
    y = pd.get_dummies(df['label'], drop_first=True).values.ravel()

    model = MultinomialNB()
    model.fit(X, y)
    return model, cv

# --- Streamlit UI ---
st.title("Email/SMS Spam Classifier")

st.markdown(
    """
    This app trains a simple CountVectorizer + MultinomialNB on `spam.csv` (v1/v2).
    Training runs once and is cached. If you'd rather ship pre-trained pickles,
    generate them offline and load instead of training here.
    """
)

# Option: load + train (runs lazily & cached)
with st.expander("Training & dataset controls"):
    st.write("If you upload your own CSV (columns `v1`/`v2` or `label`/`message`), the model will train on it.")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded, encoding='latin-1', low_memory=False)
        st.success("Uploaded dataset loaded (will be used to train).")
    else:
        try:
            df = load_dataset("spam.csv")
            st.write(f"Loaded default dataset with {len(df)} rows.")
        except Exception as e:
            st.error(f"Could not load default dataset: {e}")
            df = None

    if df is not None:
        if st.button("Train model now"):
            with st.spinner("Training model (this runs once and is cached)..."):
                try:
                    model, vectorizer = train_model_from_df(df)
                    st.success("Model trained and cached.")
                except Exception as e:
                    st.error(f"Training failed: {e}")

# Try to get cached model/vectorizer; if not trained yet, train on default dataset (if available)
model = None
vectorizer = None
try:
    # If user uploaded dataset and pressed Train, cached resource exists.
    if 'df' in locals() and isinstance(df, pd.DataFrame):
        model, vectorizer = train_model_from_df(df)
    else:
        # try default file
        default_df = load_dataset("spam.csv")  # cached
        model, vectorizer = train_model_from_df(default_df)
except Exception as e:
    st.warning("Model not available yet. Train or upload dataset in the 'Training & dataset controls' section.")
    # continue: we'll still allow single-message transform/prediction if user has pickles.

# Prediction UI
input_sms = st.text_input("Enter the message to classify")

predict_button = st.button("Predict")
if predict_button:
    if not input_sms:
        st.error("Please enter a message to predict.")
    else:
        transformed_sms = transform_text(input_sms)  # returns a string
        st.write("Transformed text:", transformed_sms)

        # If we have a trained model/vectorizer, use them
        if model is not None and vectorizer is not None:
            vector_input = vectorizer.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
        else:
            st.error("No model available. Train the model in the 'Training & dataset controls' section or load pre-trained pickles.")
