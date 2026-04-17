import pandas as pd
import numpy as np
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Config ---
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# --- 1. DATA LOADING ---
@st.cache_resource
def load_data():
    # Fetching Heart Disease dataset from OpenML (UCI)
    df = fetch_openml(name='heart-disease', version=1, as_frame=True)['frame']
    return df

df = load_data()

# --- 2. DATA PREPROCESSING ---
# Handling missing values and converting types
df = df.dropna()
# Target column is usually 'num' or 'target' (0 = No Disease, 1,2,3,4 = Disease)
# We convert it to 0 (No) and 1 (Yes)
if 'num' in df.columns:
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop('num', axis=1)
elif 'target' in df.columns:
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Separating X and y
X = df.drop('target', axis=1)
y = df['target']

# One-hot encoding for categorical variables
X = pd.get_dummies(X)

# Splitting Data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data (Important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to dataframe to keep column names (useful for GUI feature input)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)


# --- 3. MODEL TRAINING ---
@st.cache_resource
def train_models():
    # Model 1: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled_df, y_train)
    
    # Model 2: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled_df, y_train)
    
    return rf_model, lr_model

rf_model, lr_model = train_models()

# Predictions
rf_pred = rf_model.predict(X_test_scaled_df)
lr_pred = lr_model.predict(X_test_scaled_df)

# Accuracies
rf_acc = accuracy_score(y_test, rf_pred)
lr_acc = accuracy_score(y_test, lr_pred)


# --- 4. GUI DESIGN ---
st.title("❤️ ML Lab Project: Heart Disease Prediction")
st.markdown("### Comparing **Random Forest** vs **Logistic Regression**")

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Make a Prediction","Dataset Preview","Model Comparison"])

# --- TAB 1: Dataset ---
if menu == "Dataset Preview":
    st.subheader("📊 Heart Disease Dataset")
    st.write("Total Records:", df.shape[0])
    st.dataframe(df.head(10))

# --- TAB 2: Model Comparison ---
if menu == "Model Comparison":
    st.subheader("📈 Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Random Forest Accuracy", value=f"{rf_acc * 100:.2f}%")
        st.text("Classification Report:")
        st.text(classification_report(y_test, rf_pred))
        
    with col2:
        st.metric(label="Logistic Regression Accuracy", value=f"{lr_acc * 100:.2f}%")
        st.text("Classification Report:")
        st.text(classification_report(y_test, lr_pred))
        
    # Confusion Matrices
    st.subheader("Confusion Matrices")
    col3, col4 = st.columns(2)
    
    with col3:
        fig1, ax1 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title("Random Forest")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        st.pyplot(fig1)
        
    with col4:
        fig2, ax2 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, fmt='d', cmap='Greens', ax=ax2)
        ax2.set_title("Logistic Regression")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)
    # Ye code Model Comparison wale section ke end mein add kar dein
    st.subheader("🧠 What did the AI learn? (Feature Importance)")
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    fig_imp, ax_imp = plt.subplots()
    importances.plot(kind='barh', ax=ax_imp, color='lightcoral')
    ax_imp.set_title("Top 10 Features affecting Heart Disease")
    st.pyplot(fig_imp)  

# --- TAB 3: Prediction GUI (UPGRADED) ---
if menu == "Make a Prediction":
    st.subheader("🩺 Advanced AI Health Checker")
    
    # Better layout using columns
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        age = st.number_input("Age", min_value=1, max_value=100, value=45)
        sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    with col_b:
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    with col_c:
        chol = st.number_input("Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 (1=True, 0=False)", [0, 1])
        
    thalach = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    
    if st.button("🔬 Run AI Analysis", use_container_width=True):
        # 1. Adding Loading Animation
        with st.spinner('🧠 AI Model is analyzing patient data...'):
            import time
            time.sleep(2) # 2 second ka artificial loading
            
        # 2. Preparing Data
        input_data = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
        if 'age' in input_data.columns: input_data['age'] = age
        if 'sex' in input_data.columns: input_data['sex'] = sex
        if 'cp' in input_data.columns: input_data['cp'] = cp
        if 'trestbps' in input_data.columns: input_data['trestbps'] = trestbps
        if 'chol' in input_data.columns: input_data['chol'] = chol
        if 'fbs' in input_data.columns: input_data['fbs'] = fbs
        if 'thalach' in input_data.columns: input_data['thalach'] = thalach
        
        input_scaled = scaler.transform(input_data)
        
        # 3. Getting Probabilities (Percentage) instead of 0/1
        rf_proba = rf_model.predict_proba(input_scaled)[0][1] * 100  # Risk percentage
        lr_proba = lr_model.predict_proba(input_scaled)[0][1] * 100
        
        st.success("Analysis Complete!")
        
        # 4. Displaying Results beautifully
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric(label="Random Forest Risk Score", value=f"{rf_proba:.1f}%")
            st.progress(rf_proba / 100) # Progress bar
            if rf_proba > 50:
                st.error("🚨 High Risk Detected!")
            else:
                st.success("✅ Patient is Safe")
                
        with col_res2:
            st.metric(label="Logistic Regression Risk Score", value=f"{lr_proba:.1f}%")
            st.progress(lr_proba / 100) # Progress bar
            if lr_proba > 50:
                st.error("🚨 High Risk Detected!")
            else:
                st.success("✅ Patient is Safe")
