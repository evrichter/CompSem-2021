# 📊 Assignment 3 — Twitter Polarity Classification (SemEval 2016)  

Final project from the *Computational Semantics* course at **Universitat Pompeu Fabra (WS 2021/22)**.  
Task: **SemEval-2016 Task 4, Subtask A** → classify tweets as **positive, negative, or neutral**.  

---

## 🔎 Overview  

- **Data:** SemEval-2013 train/test (training) + SemEval-2016 test (evaluation)  
- **Model:** Fine-tuned **BERT (bert-base-cased)** with dropout & Adam optimizer  
- **Results:**  
  - SemEval-2013: **~77% F1**  
  - SemEval-2016: **~65% F1**  
- **Error sources:** sarcasm, negation, slang, neutral class confusion  

Details in the [report](./SemEval-2013_16%20Task%204%2C%20Sentiment%20Analysis%20in%20Twitter.pdf).  

---

## 📂 Project Structure  

```text
├── MultiClassASS3.ipynb          # BERT fine-tuning & evaluation
├── finetuned_BERT_epoch_1/       # Saved model checkpoint
├── semeval-2013-train.csv        # Training data
├── semeval-2013-test.csv         # Dev/test data
├── twitter-2016test-A.txt        # Official test set
├── SemEval-2013_16 Task 4.pdf    # Report
└── README.md
```

---

## ⚙️ Environment  

```text
python 3.8.5
torch 1.7.1
transformers 4.2.2
scikit-learn 0.24.1
numpy 1.19.4
```

