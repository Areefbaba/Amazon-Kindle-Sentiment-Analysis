import re
import pickle
from pathlib import Path

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"
TFIDF_PATH = BASE_DIR / "tfidf.pkl"

st.set_page_config(page_title="Amazon Kindle Sentiment Analysis", page_icon="📚", layout="wide")

@st.cache_resource
def load_nlp():
    try:
        sw = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        sw = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    try:
        lemmatizer.lemmatize("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    return sw, lemmatizer

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not TFIDF_PATH.exists():
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

STOP_WORDS, LEMMATIZER = load_nlp()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return " ".join(
        LEMMATIZER.lemmatize(word)
        for word in text.split()
        if word not in STOP_WORDS
    )

def train_from_dataset():
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    data_path = BASE_DIR / "all_kindle_review.csv"
    if not data_path.exists():
        raise FileNotFoundError("all_kindle_review.csv not found.")
    df = pd.read_csv(data_path, engine="python")

    def label(rating):
        if rating >= 4:
            return "Positive"
        if rating == 3:
            return "Neutral"
        return "Negative"

    df["sentiment"] = df["rating"].apply(label)
    df["clean_review"] = df["reviewText"].fillna("").apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_review"])
    model = LinearSVC()
    model.fit(X, df["sentiment"])

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    return model, vectorizer

model, vectorizer = load_artifacts()

st.title("📚 Amazon Kindle Review Sentiment Analysis")
st.caption("NLP pipeline from the project notebook: preprocessing → TF-IDF → Linear SVM")

if model is None or vectorizer is None:
    st.warning("Model artifacts are not in the repository. The first prediction will train the model from the included dataset.")

tab1, tab2 = st.tabs(["🔎 Predict", "📊 Model Info"])

with tab1:
    review = st.text_area("Enter an Amazon Kindle review", height=180, placeholder="This book is amazing and very informative.")
    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if not review.strip():
            st.error("Please enter a review.")
        else:
            try:
                if model is None or vectorizer is None:
                    with st.spinner("Training model on first run..."):
                        model, vectorizer = train_from_dataset()
                cleaned = clean_text(review)
                prediction = model.predict(vectorizer.transform([cleaned]))[0]
                if prediction == "Positive":
                    st.success("😊 Positive sentiment")
                elif prediction == "Negative":
                    st.error("😞 Negative sentiment")
                else:
                    st.info("😐 Neutral sentiment")
                with st.expander("Preprocessed text"):
                    st.code(cleaned or "(empty after preprocessing)")
            except Exception as exc:
                st.exception(exc)

with tab2:
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset", "12,000+ reviews")
    c2.metric("Features", "TF-IDF, 5,000 max")
    c3.metric("Classifier", "Linear SVM")
    st.markdown("### Sentiment mapping")
    st.write("Rating 4–5 → Positive")
    st.write("Rating 3 → Neutral")
    st.write("Rating 1–2 → Negative")
    st.info("The app follows the current notebook. The README describes a different labeling/model choice.")