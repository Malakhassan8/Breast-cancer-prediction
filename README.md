# 🔬 CancerLens — Breast Cancer Classifier

A Streamlit web app for classifying breast tumors as **Benign** or **Malignant** using 4 machine learning algorithms trained on the Wisconsin Breast Cancer Dataset.

## Features
- 📄 Upload PDF/TXT lab reports — auto-fills feature values
- 🎚️ Manual sliders for all 9 clinical features
- 🤖 4 algorithms: Decision Tree, Naive Bayes, Neural Network, Rule Induction
- 📊 Model comparison dashboard with metrics
- 📖 Rule Induction IF-THEN rules display
- 🎨 Dark medical-grade UI

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path to `app.py`
5. Click Deploy!

## Dataset
[UCI Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29) — loaded automatically at runtime.

## ⚠️ Disclaimer
For educational purposes only. Not a medical diagnosis tool.
