# Amazon-Kindle-Sentiment-Analysis

## Overview

This project is an NLP-based sentiment analysis system developed using Amazon Kindle Book Reviews. The system analyzes customer reviews and predicts whether the sentiment is Positive or Negative using Machine Learning techniques.

The project demonstrates the complete NLP pipeline including data preprocessing, feature extraction, model training, evaluation, and deployment readiness.

---

## Problem Statement

Online platforms receive thousands of customer reviews daily. Manually analyzing customer sentiment is time-consuming and inefficient.
This project automates sentiment classification by applying Natural Language Processing (NLP) and Machine Learning techniques to customer reviews.

---

## Dataset

**Dataset:** Amazon Kindle Store Reviews Dataset

### Dataset Statistics

* Total Reviews: 12,000+
* Features: 11
* Review Ratings: 1 to 5 Stars
* Source: Amazon Kindle Store Reviews

### Key Columns
* reviewText
* summary
* rating
* reviewerID
* reviewerName

---

## Project Workflow

### 1. Data Collection

* Loaded Kindle Review Dataset
* Performed dataset inspection
* Checked missing values and duplicates

### 2. Data Preprocessing

* Lowercase Conversion
* URL Removal
* Punctuation Removal
* Stopword Removal
* Lemmatization
* Text Cleaning

### 3. Sentiment Label Creation

Ratings were converted into sentiment labels:

| Rating | Sentiment |
| ------ | --------- |
| 4,5    | Positive  |
| 1,2    | Negative  |

3-star reviews were removed to reduce ambiguity and improve classification performance.

### 4. Feature Engineering

Combined:

* Review Summary
* Review Text

Additional Features:

* Review Length
* Word Count

### 5. Text Vectorization

TF-IDF Vectorization was applied with:

* Unigrams
* Bigrams
* Maximum Features = 15000

### 6. Model Training

The following models were trained and evaluated:

* Logistic Regression
* Multinomial Naive Bayes
* Linear SVM
* Random Forest

---
## Model Performance

The following machine learning models were trained and evaluated on the Amazon Kindle Review dataset:

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 75.54%   |
| Naive Bayes         | 73.29%   |
| Linear SVM          | 73.00%   |
| Random Forest       | 70.92%   |

### Best Performing Model

**Logistic Regression**

* Accuracy: 75.54%
* Selected as the final model due to its superior performance on the dataset.

### Evaluation Metrics

The models were evaluated using:

* Accuracy Score
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Observations

* Logistic Regression achieved the highest classification accuracy among all tested models.
* TF-IDF feature extraction significantly improved model performance compared to basic Bag-of-Words representations.
* The dataset contains diverse review styles and neutral opinions, making sentiment classification a challenging NLP task.

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* NLTK
* Matplotlib
* Seaborn

### NLP Techniques

* Text Cleaning
* Stopword Removal
* Lemmatization
* TF-IDF Vectorization

---
## Business Insights

The analysis revealed:

* Positive reviews frequently contain words such as:

  * excellent
  * useful
  * informative
  * recommended

* Negative reviews commonly include:

  * boring
  * disappointing
  * poor
  * waste

These insights can help publishers understand customer preferences and improve content quality.

---

## Future Improvements

* Deep Learning Models (LSTM, GRU)
* BERT-based Sentiment Classification
* Aspect-Based Sentiment Analysis
* Real-time Streamlit Deployment
* Explainable AI for sentiment prediction

---

## Author
Areef Baba
