# Amazon Kindle Sentiment Analysis

A machine learning project for analyzing customer sentiment from Amazon Kindle reviews using Natural Language Processing (NLP), TF-IDF feature extraction, and a Linear Support Vector Machine (Linear SVM) classifier. The project includes data preprocessing, model training, evaluation, and an interactive Streamlit web application for sentiment prediction.

---

## Project Overview

This project classifies Amazon Kindle product reviews into three sentiment categories:

* **Positive** → Ratings **4–5**
* **Neutral** → Rating **3**
* **Negative** → Ratings **1–2**

The system processes raw review text, converts it into numerical features using TF-IDF, and predicts sentiment using a Linear SVM model.

---

## Dataset

The project uses the **Amazon Kindle Reviews Dataset**, containing approximately:

* **12,000+ reviews**
* **11 columns**
* Customer ratings
* Review text
* Review summaries
* Product metadata

### Important Note

Although the dataset contains a `summary` column, the current implementation uses only:

```python
reviewText
```

for training and prediction.

---

## Sentiment Label Creation

Ratings are converted into sentiment labels as follows:

| Rating | Sentiment |
| ------ | --------- |
| 1–2    | Negative  |
| 3      | Neutral   |
| 4–5    | Positive  |

Example implementation:

```python
def create_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"
```

---

## Text Preprocessing

The following preprocessing steps are applied:

* Convert text to lowercase
* Remove URLs
* Remove punctuation
* Remove special characters
* Remove stopwords
* Perform lemmatization
* Remove extra spaces

Example:

### Input

```text
This Kindle is AMAZING! I love reading books on it.
```

### Processed Output

```text
kindle amazing love reading book
```

---

## Feature Engineering

Text is converted into numerical vectors using:

```python
TfidfVectorizer(max_features=5000)
```

### TF-IDF Settings

| Parameter        | Value                        |
| ---------------- | ---------------------------- |
| Method           | TF-IDF                       |
| Maximum Features | 5000                         |
| N-grams          | Default                      |
| Stopwords        | Removed during preprocessing |

---

## Machine Learning Models Evaluated

The notebook trains and evaluates multiple models:

* Logistic Regression
* Multinomial Naive Bayes
* Random Forest
* Linear Support Vector Machine (Linear SVM)

---

## Final Selected Model

The current notebook selects:

```python
LinearSVC()
```

as the final prediction model.

This model is also used inside:

```text
app.py
```

making the Streamlit application consistent with the notebook implementation.

---

## Training Workflow

### Notebook Workflow

```text
Dataset
   ↓
Text Cleaning
   ↓
Sentiment Labeling
   ↓
TF-IDF (5000 features)
   ↓
Train/Test Split
   ↓
Train Multiple Models
   ↓
Evaluate Models
   ↓
Select Linear SVM
   ↓
Prediction
```

---

## Streamlit Application

The project includes an interactive Streamlit app where users can enter custom review text and receive sentiment predictions.

### Features

* Real-time prediction
* Three-class sentiment output
* Same preprocessing as notebook
* Same TF-IDF configuration
* Same Linear SVM algorithm

Run locally:

```bash
streamlit run app.py
```

---

## Project Structure

```text
Amazon-Kindle-Sentiment-Analysis/
│
├── notebook.ipynb
├── app.py
├── requirements.txt
├── README.md
├── kindle_reviews.csv
└── assets/
```

---

## Technologies Used

### Programming

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* NLTK
* Streamlit
* Matplotlib
* Seaborn

---

## Known Limitations

### 1. TF-IDF Before Train/Test Split

In the current notebook:

```python
tfidf.fit_transform()
```

is applied before splitting the dataset.

This can introduce minor data leakage because vocabulary information from the test set becomes available during training.

A more production-ready workflow would be:

```text
Split Data
   ↓
Fit TF-IDF on Training Data
   ↓
Transform Train/Test
   ↓
Train Model
```

---

### 2. Model Retraining in Streamlit App

The current `app.py` retrains the model from the dataset if saved model files are not available.

Workflow:

```text
Dataset
   ↓
Training
   ↓
Prediction
```

A production deployment would instead use:

```text
Saved TF-IDF
Saved Model
   ↓
Prediction
```

using:

```text
tfidf.pkl
sentiment_model.pkl
```

---

### 3. Summary Column Not Used

The dataset includes:

* reviewText
* summary

Currently, only:

```text
reviewText
```

is used for training.

Combining both fields may improve model performance.

---

## Future Improvements

* Save trained models (`.pkl`)
* Hyperparameter tuning
* Cross-validation
* Use review summaries
* Deep learning models (LSTM, BERT)
* Transformer-based sentiment analysis
* Model deployment on cloud platforms

---

## Example Predictions

| Review                                 | Prediction |
| -------------------------------------- | ---------- |
| "Amazing product, highly recommended!" | Positive   |
| "Average reading experience."          | Neutral    |
| "Battery life is terrible."            | Negative   |

---
