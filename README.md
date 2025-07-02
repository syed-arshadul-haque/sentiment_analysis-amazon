# NLP Workflow for Text Data: Preprocessing, Classification, and Topic Modeling

This project demonstrates a comprehensive workflow for performing Natural Language Processing (NLP) on text data. It includes data preprocessing, sentiment classification using traditional ML models and transformer-based models, and topic modeling using LDA and NMF.

## 📁 Project Structure

- **Chapter 1: Text Preprocessing**
  - Removal of punctuation, numbers, and stop words
  - Conversion to lowercase
  - Lemmatization and spelling correction using `spaCy` and `pyspellchecker`

- **Chapter 2: Text Classification using Traditional ML**
  - Bag-of-Words feature extraction using `CountVectorizer`
  - Classification models: Random Forest, Bagging, KNN, AdaBoost
  - Hyperparameter tuning with `GridSearchCV`
  - Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC Curve

- **Chapter 3: Classification with Transformers**
  - Fine-tuned `BERT-base` and `RoBERTa-base` using Hugging Face Transformers
  - Data tokenization and encoding
  - Training with PyTorch `Trainer`
  - Evaluation with confusion matrix and ROC AUC

- **Chapter 4: Topic Modeling**
  - Latent Dirichlet Allocation (LDA) using `gensim`
  - Non-negative Matrix Factorization (NMF) using `sklearn`
  - Visualization with `pyLDAvis`

## 📊 Performance Summary

| Model      | Accuracy |
|------------|----------|
| RoBERTa    | 94%      |
| BERT       | 92%      |
| Random Forest | 82.8% |
| Bagging    | 80%      |
| KNN        | 77.2%    |
| AdaBoost   | 76.8%    |

## 🧠 Topic Modeling Summary

Extracted and interpreted 10 distinct topics using both LDA and NMF, capturing insights into product reviews, sentiment, and user experiences with mobile-related products.

## 🛠️ Technologies Used

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- NLTK, spaCy
- Hugging Face Transformers
- Gensim, pyLDAvis
- Matplotlib, Seaborn

## 🧑‍💻 Author

- **Student ID**: UP2280648  
- **Submission Date**: January 20, 2025  
- **Module**: Intelligent Data and Text Analytics (M33147)

---

Feel free to fork, contribute, or raise an issue for feedback!
