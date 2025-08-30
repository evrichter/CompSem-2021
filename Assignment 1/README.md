# 📊 Assignment 1 — Sentiment Classification  

This is the first assignment from the **Computational Semantics** course at **Universitat Pompeu Fabra (WS 2021/22)**.  
The task was to implement a **binary sentiment classifier** and perform a brief error analysis.  

---

## 🔎 Task Overview  

- Preprocessing: stopword & punctuation removal, contraction expansion, lemmatization  
- Features: **CountVectorizer** (14k+ unique tokens)  
- Classifier: **Logistic Regression** (scikit-learn)  
- Evaluation: ~76% accuracy on the test set  

More details can be found in the [Report](./Assignment_1_Report_Error_Analysis.pdf)



---

## 📂 Project Structure  

```text
├── sentiment_classifier.py   # Logistic regression classifier
├── vectorizer.sav            # Saved CountVectorizer
├── final.csv                 # Dataset (train/val/test split)
├── Assignment 1_Report.pdf   # Report with error analysis
└── README.md
```

---

## ⚙️ Environment  

```text
Requirements:
  - python 3.8.5
  - scikit-learn 0.24.1
  - nltk 3.5

System:
  - Ubuntu 20.04.1 LTS (64-bit)
```

