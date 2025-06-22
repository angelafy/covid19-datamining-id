# 📊 COVID-19 Data Mining: Clustering & Classification

This project analyzes COVID-19 data in Indonesia from 2022 using **K-Means Clustering** and **Logistic Regression**.  
The goal is to cluster provinces based on total cases and deaths, and to classify the risk level (**Low**, **Medium**, **High**) using logistic regression.

Built using **Python** and **Streamlit**, the app provides an interactive dashboard with several key features.

---

## 🚀 Features

- 📌 **Elbow Method Visualization** to determine the optimal number of clusters  
- 📊 **K-Means Clustering** results visualized with scatter plots and tabular outputs  
- 🧠 **Classification Model** using Logistic Regression with evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- ⚙️ **Data Preprocessing**:
  - Normalization using StandardScaler  
  - Categorical encoding using LabelEncoder  
  - Imbalanced class handling with **SMOTE**
- 🔎 **Filtering Tools**:
  - Filter data by province name (case-insensitive search)
  - Display filtered clustering results

---

## 🧰 Library

- `Python`
- `Pandas`, `NumPy`
- `Scikit-Learn`, `Imbalanced-Learn`
- `Matplotlib`, `Seaborn`
- `Streamlit`

---

## 📁 Dataset

- Format: CSV  
- Contains COVID-19 cases and deaths per province in Indonesia (2022)

> **Note**: Make sure to include the dataset file (`data.csv`) in the root directory when running the app.

---

## ▶️ Run the App

```bash
streamlit run app.py
