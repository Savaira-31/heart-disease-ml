# 💓 Heart Disease Prediction using Machine Learning

An advanced AI-powered web application designed to predict the risk of heart disease based on clinical patient data.  
This project implements a complete Machine Learning pipeline, compares multiple models, and provides an interactive prediction system.

---

## 🧠 Project Overview

Heart disease is one of the leading causes of death worldwide.  
This project aims to assist early diagnosis by using machine learning models to predict whether a patient is at risk of heart disease based on clinical features.

The system is trained on a real-world UCI dataset and evaluates multiple algorithms to identify the best-performing model.

---

## 🧪 Machine Learning Models Used

- Logistic Regression (Baseline Model)
- Random Forest Classifier (Ensemble Model)
- XGBoost Classifier (Best Performing Model)

---

## 🚀 Key Features

- 🔍 **3 Model Comparison**
  Evaluates Logistic Regression, Random Forest, and XGBoost

- 🧹 **Real-world Data Handling**
  Handles missing values using median (numerical) and mode (categorical) imputation

- 🔤 **Feature Encoding**
  Label Encoding applied for categorical variables

- 📊 **Model Evaluation**
  Accuracy comparison, confusion matrix, and classification report

- 📈 **Risk Prediction Output**
  Provides probability-based prediction instead of simple binary output

---

## 🧠 Machine Learning Pipeline

1. 📥 Data Collection  
   - UCI Heart Disease Dataset (920 records)

2. 🧹 Data Preprocessing  
   - Missing value treatment  
   - Data cleaning and transformation

3. 🔤 Feature Encoding  
   - Label Encoding for categorical variables

4. 🧠 Model Training  
   - Logistic Regression  
   - Random Forest  
   - XGBoost

5. 📊 Model Evaluation  
   - Accuracy comparison  
   - Confusion matrix  
   - Classification report

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn  
- **Deployment:** Streamlit (optional web app)

---

## 📊 Results

- Logistic Regression Accuracy: ~78%  
- Random Forest Accuracy: ~83%  
- XGBoost Accuracy: ~86% (Best Model)

---

## 🌐 Live Demo

https://savaira-31-heart-disease-ml-app-kaqw9l.streamlit.app/

---

## 👩‍💻 Author

Savaira Majeed  
DSAI231103031
