import pandas as pd
import numpy as np
import streamlit as st
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

@st.cache_resource
def load_models():
    lr = joblib.load('lr_model.pkl')
    rf = joblib.load('rf_model.pkl')
    xgb = joblib.load('xgb_model.pkl')
    encoders = joblib.load('encoders.pkl')
    cols = joblib.load('columns.pkl')
    return lr, rf, xgb, encoders, cols

try:
    lr_model, rf_model, xgb_model, encoders, model_columns = load_models()
except:
    st.error("Error: Models load nahi hue. Pehle 'train.py' run karein.")
    st.stop()

# Data for graphs
df = pd.read_csv('heart_disease_uci.csv')
df['target'] = (df['num'] > 0).astype(int)
df = df.drop(['id', 'dataset', 'num'], axis=1)

# MISSING VALUES FILL KARNA (Train mein jo kiya tha wo yahan bhi kiya)
num_cols = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = ['fbs', 'restecg', 'exang', 'slope', 'thal']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

y_full = df['target']
X_full = df.drop('target', axis=1)

for col in encoders.keys():
    X_full[col] = encoders[col].transform(X_full[col].astype(str))

_, X_test, _, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

pred_lr = lr_model.predict(X_test)
pred_rf = rf_model.predict(X_test)
pred_xgb = xgb_model.predict(X_test)

acc_lr = accuracy_score(y_test, pred_lr)
acc_rf = accuracy_score(y_test, pred_rf)
acc_xgb = accuracy_score(y_test, pred_xgb)

# GUI
st.title("❤️ ML Lab Project: Heart Disease Prediction")
st.markdown("### Comparing **Logistic Regression**, **Random Forest** & **XGBoost**")

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Make a Prediction", "Model Comparison", "Dataset Preview"])

if menu == "Dataset Preview":
    st.subheader("📊 Heart Disease UCI Dataset")
    st.write("Total Records:", len(df))
    st.dataframe(df.head(5))

if menu == "Model Comparison":
    st.subheader("📈 Model Performance Comparison")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Logistic Regression", f"{acc_lr*100:.2f}%")
        st.text(classification_report(y_test, pred_lr))
    with col2:
        st.metric("Random Forest", f"{acc_rf*100:.2f}%")
        st.text(classification_report(y_test, pred_rf))
    with col3:
        st.metric("XGBoost", f"{acc_xgb*100:.2f}%")
        st.text(classification_report(y_test, pred_xgb))

    st.subheader("Confusion Matrices")
    cols_cm = st.columns(3)
    models_eval = [("Logistic Regression", pred_lr), ("Random Forest", pred_rf), ("XGBoost", pred_xgb)]
    colors = ["Blues", "Greens", "Oranges"]
    
    for i, (name, pred) in enumerate(models_eval):
        with cols_cm[i]:
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap=colors[i], ax=ax)
            ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig)

    st.subheader("📊 Accuracy Bar Chart")
    fig_bar, ax_bar = plt.subplots()
    sns.barplot(x=["Logistic Reg.", "Random Forest", "XGBoost"], y=[acc_lr, acc_rf, acc_xgb], ax=ax_bar)
    ax_bar.set_ylim(0, 1)
    for i, v in enumerate([acc_lr, acc_rf, acc_xgb]):
        ax_bar.text(i, v + 0.02, f"{v*100:.1f}%", ha='center')
    st.pyplot(fig_bar)

if menu == "Make a Prediction":
    st.subheader("🩺 Advanced AI Health Checker")
    
    # DYNAMIC DROPDOWNS: Yeh encoders se exact words uthayenge (Male/Female etc)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        age = st.number_input("Age", 1, 100, 45)
        # Encoders se classes utha rahe hain
        sex_options = list(encoders['sex'].classes_)
        sex = st.selectbox("Sex", sex_options)
        
    with col_b:
        cp_options = list(encoders['cp'].classes_)
        cp = st.selectbox("Chest Pain Type", cp_options)
        trestbps = st.number_input("Blood Pressure (mm Hg)", 80, 200, 120)
        
    with col_c:
        chol = st.number_input("Cholestoral (mg/dl)", 100, 600, 200)
        fbs_options = list(encoders['fbs'].classes_)
        fbs = st.selectbox("Fasting Blood Sugar", fbs_options)
        
    col_d, col_e, col_f = st.columns(3)
    with col_d:
        restecg_options = list(encoders['restecg'].classes_)
        restecg = st.selectbox("Resting ECG", restecg_options)
        thalch = st.number_input("Max Heart Rate (thalch)", 60, 220, 150)
        
    with col_e:
        exang_options = list(encoders['exang'].classes_)
        exang = st.selectbox("Exercise Induced Angina", exang_options)
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 0.0)
        
    with col_f:
        slope_options = list(encoders['slope'].classes_)
        slope = st.selectbox("Slope", slope_options)
        ca = st.number_input("Major Vessels (ca)", 0, 4, 0)
        thal_options = list(encoders['thal'].classes_)
        thal = st.selectbox("Thalassemia (thal)", thal_options)

    if st.button("🔬 Run AI Analysis", use_container_width=True):
        with st.spinner('🧠 AI Models analyzing...'):
            time.sleep(2)
        
        # GUI inputs ko exact Notebook jaisa format mein dalein
        input_dict = {
            'age': float(age),
            'sex': sex,
            'cp': cp,
            'trestbps': float(trestbps),
            'chol': float(chol),
            'fbs': fbs,
            'restecg': restecg,
            'thalch': float(thalch),
            'exang': exang,
            'oldpeak': float(oldpeak),
            'slope': slope,
            'ca': float(ca),
            'thal': thal
        }
        
        input_df = pd.DataFrame([input_dict])
        
        # SAVED ENCODERS SE EXACTLY WAHI NUMBERS BANAYENGE JO NOTEBOOK NE BANAYE THE
        for col in encoders.keys():
            input_df[col] = encoders[col].transform(input_df[col].astype(str))
            
        # Order match karna
        input_df = input_df[model_columns]
        
        # Probabilities
        
        prob_lr = float(lr_model.predict_proba(input_df)[0][1] * 100)
        prob_rf = float(rf_model.predict_proba(input_df)[0][1] * 100)
        prob_xgb = float(xgb_model.predict_proba(input_df)[0][1] * 100)
        
        st.success("Analysis Complete!")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Logistic Reg Risk", f"{prob_lr:.1f}%")
            st.progress(prob_lr / 100)
            if prob_lr > 50: st.error("🚨 High Risk")
            else: st.success("✅ Safe")
            
        with c2:
            st.metric("Random Forest Risk", f"{prob_rf:.1f}%")
            st.progress(prob_rf / 100)
            if prob_rf > 50: st.error("🚨 High Risk")
            else: st.success("✅ Safe")
            
        with c3:
            st.metric("XGBoost Risk", f"{prob_xgb:.1f}%")
            st.progress(prob_xgb / 100)
            if prob_xgb > 50: st.error("🚨 High Risk")
            else: st.success("✅ Safe")
