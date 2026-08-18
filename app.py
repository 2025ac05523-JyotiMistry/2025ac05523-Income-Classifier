import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_curve
import os

st.set_page_config(page_title="Income Classifier", layout="wide")

st.title("💰 Income Classifier")

st.markdown("""
This app predicts whether an individual's income exceeds $50,000 using 5 ML models.
""")

# Function to fix column names
def fix_column_names(df):
    """Fix common column name variations"""
    mapping = {
        'education.num': 'education-num',
        'marital.status': 'marital-status',
        'capital.gain': 'capital-gain',
        'capital.loss': 'capital-loss',
        'hours.per.week': 'hours-per-week',
        'native.count': 'native-country',
        'workplace': 'workclass'
    }
    df = df.copy()
    for old, new in mapping.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    return df

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
        except Exception as e:
            st.error(f"Failed to load {name}: {str(e)}")
            return None, None, None
    
    try:
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor_data = pickle.load(f)
            preprocessor = preprocessor_data['preprocessor']
            expected_columns = preprocessor_data['expected_columns']
    except Exception as e:
        st.error(f"Failed to load preprocessor: {str(e)}")
        return None, None, None
    
    return models, preprocessor, expected_columns

models, preprocessor, expected_columns = load_models()

if models is None or preprocessor is None or expected_columns is None:
    st.error("❌ Failed to load models. Please check the models directory.")
    st.stop()

st.success("✅ Models loaded successfully!")

st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose a model", list(models.keys()))

st.header("Upload Test Data")
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)
        st.success(f"Data loaded: {test_data.shape[0]} rows, {test_data.shape[1]} columns")
        
        st.subheader("Data Preview")
        st.dataframe(test_data.head())
        
        if st.button("Make Predictions"):
            try:
                # Fix column names first
                X_test = fix_column_names(test_data)
                
                # Remove target if exists
                if 'income' in X_test.columns:
                    y_test = X_test['income']
                    X_test = X_test.drop('income', axis=1)
                else:
                    y_test = None
                
                # Ensure all expected columns exist
                for col in expected_columns:
                    if col not in X_test.columns:
                        if col in ['workclass', 'education', 'marital-status', 'occupation', 
                                  'relationship', 'race', 'sex', 'native-country']:
                            X_test[col] = 'Unknown'
                        else:
                            X_test[col] = 0
                
                # Select only expected columns in correct order
                X_test = X_test[expected_columns]
                
                # Convert numeric columns
                numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
                for col in numeric_cols:
                    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
                
                # Convert categorical columns to string
                cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                           'relationship', 'race', 'sex', 'native-country']
                for col in cat_cols:
                    X_test[col] = X_test[col].astype(str).replace(' ?', 'Unknown').fillna('Unknown')
                
                # Now transform
                X_test_processed = preprocessor.transform(X_test)
                
                # Make predictions
                model = models[selected_model]
                y_pred = model.predict(X_test_processed)
                y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
                
                # Display predictions
                st.subheader("Predictions")
                results = pd.DataFrame({
                    'Prediction': ['>50K' if pred == 1 else '<=50K' for pred in y_pred],
                    'Probability (>50K)': y_pred_proba
                })
                st.dataframe(results)
                
                # Download predictions
                csv = results.to_csv(index=False)
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
                
                # If we have actual values, show metrics and confusion matrix
                if y_test is not None:
                    st.header("Model Evaluation")
                    
                    # Calculate metrics
                    metrics = {
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred),
                        'Recall': recall_score(y_test, y_pred),
                        'F1 Score': f1_score(y_test, y_pred),
                        'MCC': matthews_corrcoef(y_test, y_pred),
                        'AUC': roc_auc_score(y_test, y_pred_proba)
                    }
                    
                    # Display metrics
                    cols = st.columns(3)
                    metric_items = list(metrics.items())
                    for i, (name, value) in enumerate(metric_items):
                        cols[i % 3].metric(name, f"{value:.4f}")
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['<=50K', '>50K'],
                                yticklabels=['<=50K', '>50K'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                    plt.close()
                    
                    # ROC Curve
                    st.subheader("ROC Curve")
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(fpr, tpr, label=f'{selected_model} (AUC = {metrics["AUC"]:.3f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.legend()
                    st.pyplot(fig)
                    plt.close()
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.write("Debug - Data types:", X_test.dtypes)
                st.write("Debug - Data shape:", X_test.shape)
                st.write("Debug - Columns:", X_test.columns.tolist())
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

st.markdown("---")
st.caption("ML Assignment 2 - Income Classification")