# 🔬 CancerLens — Breast Cancer Classifier

A Streamlit web app that classifies breast tumors as **Benign** or **Malignant** using 4 machine learning algorithms trained on the Wisconsin Breast Cancer Dataset.


## 🚀 Try [Live Demo](https://breast-cancer-prediction-8v3gtkney8bkrfcslpbtft.streamlit.app/)

---

## ✨ Features

- 📄 **PDF / TXT Upload** — upload a lab report and sliders auto-fill automatically
- 🎚️ **Manual Sliders** — adjust all 9 clinical features individually (scale 1–10)
- 🤖 **4 ML Algorithms** — choose between Decision Tree, Naive Bayes, Neural Network, or Rule Induction
- ⚡ **Run All Models** — compare all 4 algorithms on the same input simultaneously
- 📊 **Verdict Banner** — shows majority decision and whether models agree or disagree
- 📉 **Feature Risk Profile** — color-coded bars showing risk level per feature
- 📖 **Rule Induction Rules** — displays human-readable IF-THEN rules that led to the decision
- 📈 **Model Comparison Dashboard** — accuracy, precision, recall and F1-score for all 4 models
- ⚡ **No file dependencies** — dataset loads and models train automatically at startup

---

## 🧠 Models

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree | 94.89% | 93.00% | 92.00% | 92.50% |
| Naive Bayes | 94.16% | 92.00% | 92.00% | 92.00% |
| Neural Network | 97.08% | 100.00% | 92.00% | 95.83% |
| Rule Induction | 96.35% | 97.87% | 92.00% | 94.85% |

### Model Configurations
| Algorithm | Configuration |
|---|---|
| Decision Tree | `criterion='entropy'`, `max_depth=4`, `random_state=0` |
| Naive Bayes | `GaussianNB` (default) |
| Neural Network | `MLPClassifier` · hidden layers `(16, 8)` · `relu` · `adam` · `max_iter=500` |
| Rule Induction | `RuleFitClassifier` · `random_state=42` |

---

## 📋 Features Used

| # | Feature | Description |
|---|---|---|
| 1 | Clump Thickness | Thickness of cell clumps (1–10) |
| 2 | Uniformity of Cell Size | Consistency of cell sizes (1–10) |
| 3 | Uniformity of Cell Shape | Consistency of cell shapes (1–10) |
| 4 | Marginal Adhesion | How well cells stick together (1–10) |
| 5 | Single Epithelial Cell Size | Size of individual epithelial cells (1–10) |
| 6 | Bare Nuclei | Nuclei not surrounded by cytoplasm (1–10) |
| 7 | Bland Chromatin | Texture of the nucleus (1–10) |
| 8 | Normal Nucleoli | Presence of nucleoli (1–10) |
| 9 | Mitoses | Rate of cell division (1–10) |

---

## 🗂️ Dataset

| Property | Detail |
|---|---|
| Name | Wisconsin Breast Cancer Dataset |
| Source | UCI Machine Learning Repository |
| Instances | 699 patients |
| Features | 9 clinical features |
| Target | Benign (0) / Malignant (1) |
| Class Distribution | ~65% Benign · ~35% Malignant |
| Train/Test Split | 80% / 20% (`random_state=40`) |
| Preprocessing | StandardScaler |

---

## 🖥️ Run Locally

```bash
git clone https://github.com/yourusername/breast-cancer-classifier
cd breast-cancer-classifier
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → set main file to `app.py`
5. Click **Deploy** — live in ~2 minutes

---

## 📁 Project Structure

```
breast-cancer-classifier/
├── app.py                 ← Main Streamlit app
├── requirements.txt       ← Python dependencies
├── README.md              ← You are here
└── Saved_Models/
    ├── decision_tree.sav
    ├── naive_bayes.sav
    ├── neural_network.sav
    ├── rule_induction.sav
    └── scaler.sav
```

---

## 💡 Key Findings

- **Neural Network** achieved the highest accuracy (97.08%) with perfect Precision (100%) — zero false positives. It never incorrectly flags a benign tumor as malignant
- **Rule Induction** (96.35%) is the most interpretable — produces human-readable IF-THEN rules doctors can validate directly
- **Decision Tree** was updated to 94.89% accuracy using `criterion='entropy'` and `max_depth=4`, outperforming the previous configuration
- **Naive Bayes** performs competitively at 94.16% despite being the simplest model — no hyperparameter tuning required
- **Bare Nuclei, Cell Size & Cell Shape** are the strongest predictors of malignancy (correlation r = 0.82)
- **Cell Size and Cell Shape** are highly correlated with each other (r = 0.91), indicating feature redundancy
- **Mitoses** is the weakest predictor (correlation r = 0.42) and rarely appears in top rules
- All models achieve identical Recall (92%) — the key differentiator is Precision and Accuracy, where Neural Network leads
- Outliers were **retained** because in medical datasets they represent real extreme clinical cases; StandardScaler was applied to control their influence

---

## 🔗 Related Project

Also check out my [**Diabetes Risk Predictor**](https://github.com/Malakhassan8/Diabetes-Prediction) — predicts diabetes risk from 8 clinical indicators with PDF upload and a visual risk gauge.

---

## ⚠️ Disclaimer

This application is for **educational purposes only** and does not constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.

---

## 👤 Author

[**Malak Hassan**](https://www.linkedin.com/in/malak-hassan-b2b984271/)

---
## Contributors
[**Rabab Mohamed**](https://github.com/Rabab-Okasha)
