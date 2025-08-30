# 📊 Assignment 2 — Lexical Semantics  

This is the second assignment from the **Computational Semantics** course at **Universitat Pompeu Fabra (WS 2021/22)**.  
The assignment consisted of several exercises on **lexical semantics, word embeddings, and semantic evaluation**.  

---

## 🔎 Task Overview  

- **Exercise 2 — Word Sense Annotation**  
  - Annotated 100 sentences with senses of the word *water* (10 synsets).  
  - Inter-annotator agreement: **84%**.  
  - Disagreements often due to context (e.g., cooking vs. chemical substance).  

- **Exercise 3 — Synonyms, Antonyms, Hypernyms & Hyponyms**  
  - Compared embeddings for different lexical relations using Word2Vec + WordNet.  
  - Result: **no significant difference** → embeddings are too coarse to model fine-grained relations.  

- **Exercise 4 — Essay Questions**  
  - Mini-essay **for distributional semantics** (advantages over sense enumeration).  
  - Mini-essay **for sense enumeration** (problems with meaning conflation in embeddings).  

- **Exercise 5 — Word Similarity & Relatedness (graded part)**  
  - Evaluated **Word2Vec** (43k vectors) vs. **FastText** (2M vectors, 300d) on the **WordSim353** dataset.  
  - FastText outperformed Word2Vec:  
    - FastText: **ρ = 0.83 (similarity), 0.74 (relatedness)**  
    - Word2Vec: **ρ = 0.77 (similarity), 0.60 (relatedness)**  
  - Embeddings capture similarity better than relatedness.  

More details are in [Exercise2-4.pdf](./Exercise2-4.pdf) and [Exercise5_Eva_Nora.pdf](./Exercise5_Eva_Nora.pdf).  

---

## 📂 Project Structure  

```text
├── ws353simrel/                 # WordSim353 dataset
├── Excercise5_wordsim.py         # Evaluation script
├── Exercise2.py                  # Word sense annotation
├── Exercise3.py                  # Synonyms/antonyms analysis
├── Exercise2-4.pdf               # Report (Exercises 2–4)
├── Exercise5_Eva_Nora.pdf        # Report (Exercise 5, graded part)
├── CSV_Fasttext_Rel:Results.csv  # Results FastText (relatedness)
├── CSV_Fasttext_Sim:Results.csv  # Results FastText (similarity)
├── CSV_W2V_Rel:Results.csv       # Results Word2Vec (relatedness)
├── CSV_W2V_Sim:Results.csv       # Results Word2Vec (similarity)
└── README.md
```

---

## ⚙️ Environment  

```text
Requirements:
  - python 3.8.5
  - gensim 4.0.1
  - numpy 1.19.4
  - scipy 1.6.2
  - nltk 3.5
```

