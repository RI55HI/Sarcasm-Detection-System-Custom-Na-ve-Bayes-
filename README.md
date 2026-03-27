# Sarcasm Detection System (Custom Naïve Bayes)

A from-scratch implementation of a Natural Language Processing (NLP) pipeline and a Multinomial/Bernoulli Naïve Bayes classifier to detect sarcasm in news headlines. 

This project was developed for the **Applied Machine Learning (COMP9060)** module at MTU.

## 📌 Project Overview
The goal is to classify news headlines into two categories:
1. **Sarcastic:** Headlines from *The Onion*.
2. **Serious:** Headlines from *HuffPost*.

The core of this project is the manual development of ML algorithms using only **NumPy** and **Pandas**, avoiding high-level libraries like `scikit-learn` for the model logic.

## 🛠️ Individual Constraints (Seed: 77319)
The project is parameterized based on student-specific seeds. My implementation follows these constraints:
- **Rare-word threshold ($t$):** 5 (Words appearing < 5 times are discarded).
- **Ablation index ($k$):** 3.
- **Laplace Smoothing ($\alpha$):** $\{0, 0.1, 1\}$.
- **Negation Rule:** Vowel-count based prefixing (e.g., "never" has 2 vowels $\rightarrow$ next 2 words prefixed with `NOT_`).
- **Feature Reduction:** Prime-Index Scheme (Only tokens at prime-numbered alphabetical positions are kept as unique features).

## 🚀 Features & Pipeline
### 1. NLP Preprocessing
- **Marker Substitution:** Preserves tonal signals like `!!!`, `???`, and `?!` by converting them to unique tokens.
- **Negation Handling:** Custom vowel-count logic to capture context within a bag-of-words model.
- **N-Grams:** Generation of unigrams and bigrams (e.g., `area_man`).
- **OOV Handling:** Implementation of `<UNK>` tokens for out-of-vocabulary words in test data.

### 2. Feature Engineering
- **Textual:** Manual implementation of Count Vectorizer (Prime-Index restricted) and TF-IDF.
- **Numeric:** Extraction of 7 structural features (Word count, uppercase ratio, punctuation frequency, etc.).
- **Data Cleaning:** IQR-based outlier detection and Clamping; Class-conditional mean imputation for missing values.

### 3. Machine Learning Models
- **Multinomial Naïve Bayes:** Best performing model ($\alpha=0.1$).
- **Bernoulli Naïve Bayes:** Comparative model for binary presence/absence.
- **Gaussian Naïve Bayes:** Used for numeric feature analysis.
- **Log-Space Math:** All probabilities computed in log-space to prevent numerical underflow.

## 📊 Key Results
| Model Configuration | Val Acc | Val Macro-F1 |
| :--- | :---: | :---: |
| Multinomial NB ($\alpha=0.1$) | **63.72%** | **0.6360** |
| Bernoulli NB ($\alpha=0.1$) | 62.55% | 0.6024 |
| MNB + Oversampling (Thr=0.5) | 61.10% | 0.5983 |
| **Final Test Set Result** | **60.43%** | **0.5945** |

### Insights
- **Sarcastic Recall:** Achieved **82%** after Random Oversampling and threshold tuning.
- **Semantic Irony:** The model excels at identifying stylistic sarcasm (The Onion tropes) but remains limited in detecting irony reliant on semantic incongruity due to the "Independence Assumption" of Naïve Bayes.

## 📂 Repository Structure
- `COMP9060_A1_roo277319.ipynb`: Full source code and interactive demo.
- `report.pdf`: Detailed technical documentation and error analysis.
- `confusion_matrix.png`: Performance visualization.
- `README.md`: Project summary.

## 📝 Author
- **Rishikesh Mahendra Dharane**
- MSc in Applied Computing, MTU.
- Student ID: R00277319

## 📜 License
This project is for academic purposes as part of the COMP9060 module.
