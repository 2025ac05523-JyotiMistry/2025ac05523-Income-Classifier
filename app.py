import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import os

st.set_page_config(page_title="Income Classifier", layout="wide")

st.title("💰 Income Classification App")
st.markdown("Predicts if income exceeds $50K using Census data")

@st.cache_resource
def load_models():
    models = {}
    model_paths = {
        'Logistic Regression': 'models/model_logistic_regression.pkl',
        'Decision Tree': 'models/model_decision_tree.pkl',
        'KNN': 'models/model_knn.pkl',
        'Naive Bayes': 'models/model_naive_bayes.pkl',
        'Random Forest': 'models/model_random_forest.pkl'
    }
    
    for name, path in model_paths.items():
        try:
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
        except:
            st.error(f"Failed to load {name}")
            return None
    
    try:
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
    except:
        st.error("Failed to load preprocessor")
        return None
    
    return models, preprocessor

models, preprocessor = load_models()
if models is None:
    st.stop()

st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose a model", list(models.keys()))

st.header("Upload Test Data")
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} rows")
        
        with st.expander("Data Preview"):
            st.dataframe(df.head())
        
        # Check if target exists
        has_target = 'income' in df.columns
        
        if has_target:
            y_true = df['income'].values
            X = df.drop('income', axis=1)
        else:
            X = df
            y_true = None
        
        # Display features
        st.write(f"Features: {', '.join(X.columns)}")
        
        if st.button("Make Predictions"):
            try:
                # Convert all columns to numeric, replace non-numeric with 0
                X_clean = X.copy()
                for col in X_clean.columns:
                    X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce').fillna(0)
                
                # Preprocess
                X_processed = preprocessor.transform(X_clean)
                
                # Predict
                model = models[selected_model]
                y_pred = model.predict(X_processed)
                y_proba = model.predict_proba(X_processed)[:, 1]
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(y_pred))
                with col2:
                    st.metric("Predicted >50K", sum(y_pred == 1))
                with col3:
                    st.metric("Predicted <=50K", sum(y_pred == 0))
                
                # Show predictions
                results = pd.DataFrame({
                    'Prediction': ['>50K' if p == 1 else '<=50K' for p in y_pred],
                    'Probability': y_proba
                })
                st.dataframe(results)
                
                # Download
                csv = results.to_csv(index=False)
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
                
                # If target exists, show metrics
                if has_target:
                    st.header("Evaluation Metrics")
                    
                    acc = accuracy_score(y_true, y_pred)
                    auc = roc_auc_score(y_true, y_proba)
                    prec = precision_score(y_true, y_pred)
                    rec = recall_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred)
                    mcc = matthews_corrcoef(y_true, y_pred)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{acc:.4f}")
                        st.metric("Precision", f"{prec:.4f}")
                    with col2:
                        st.metric("Recall", f"{rec:.4f}")
                        st.metric("F1 Score", f"{f1:.4f}")
                    with col3:
                        st.metric("AUC", f"{auc:.4f}")
                        st.metric("MCC", f"{mcc:.4f}")
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['<=50K', '>50K'],
                                yticklabels=['<=50K', '>50K'])
                    ax.set_title(f'{selected_model}')
                    st.pyplot(fig)
                    plt.close()
                    
                    # Classification Report
                    st.subheader("Classification Report")
                    from sklearn.metrics import classification_report
                    report = classification_report(y_true, y_pred, target_names=['<=50K', '>50K'])
                    st.text(report)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Try cleaning your data: ensure all columns are numeric")
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

st.markdown("---")
st.caption("ML Assignment 2 - Income Classification")