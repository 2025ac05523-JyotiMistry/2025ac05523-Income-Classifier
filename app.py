import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                            recall_score, f1_score, matthews_corrcoef,
                            confusion_matrix, roc_curve)
import os
import re

st.set_page_config(
    page_title="Income Classifier",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #A23B72;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">💰 Income Classifier</p>', unsafe_allow_html=True)

st.markdown("""
This application uses machine learning models to predict whether an individual's 
annual income exceeds $50,000 based on demographic and employment-related features.

**Dataset**: Adult Census Income Dataset (UCI)
""")

def fix_column_names(df):
    column_mapping = {
        'education.num': 'education-num',
        'education-num': 'education-num',
        'marital.status': 'marital-status',
        'marital-status': 'marital-status',
        'capital.gain': 'capital-gain',
        'capital-gain': 'capital-gain',
        'capital.loss': 'capital-loss',
        'capital-loss': 'capital-loss',
        'hours.per.week': 'hours-per-week',
        'hours-per-week': 'hours-per-week',
        'native.count': 'native-country',
        'native-country': 'native-country',
        'workplace': 'workclass',
        'workclass': 'workclass'
    }
    
    df_fixed = df.copy()
    
    for old_name, new_name in column_mapping.items():
        if old_name in df_fixed.columns:
            df_fixed = df_fixed.rename(columns={old_name: new_name})
    
    return df_fixed

def prepare_data(df, expected_columns):
    df_fixed = fix_column_names(df)
    
    missing_columns = set(expected_columns) - set(df_fixed.columns)
    
    if missing_columns:
        st.warning(f"⚠️ Missing columns: {missing_columns}")
        st.info("Attempting to fix column issues...")
        
        for col in missing_columns:
            for existing_col in df_fixed.columns:
                if col.replace('-', '').replace('_', '').lower() in existing_col.replace('-', '').replace('_', '').lower():
                    df_fixed = df_fixed.rename(columns={existing_col: col})
                    break
        
        missing_columns = set(expected_columns) - set(df_fixed.columns)
        if missing_columns:
            st.warning(f"Adding missing columns with default values: {missing_columns}")
            for col in missing_columns:
                df_fixed[col] = 0
    
    for col in expected_columns:
        if col not in df_fixed.columns:
            df_fixed[col] = 0
    
    df_final = df_fixed[expected_columns]
    
    return df_final

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
    
    if not os.path.exists('models'):
        st.error("❌ Models directory not found! Please ensure the models folder exists.")
        return None, None
    
    for model_name, model_path in model_paths.items():
        try:
            with open(model_path, 'rb') as file:
                models[model_name] = pickle.load(file)
        except FileNotFoundError:
            st.error(f"❌ Model file not found: {model_path}")
            return None, None
        except Exception as e:
            st.error(f"❌ Error loading {model_name}: {str(e)}")
            return None, None
    
    try:
        with open('models/preprocessor.pkl', 'rb') as file:
            preprocessor = pickle.load(file)
    except FileNotFoundError:
        st.error("❌ Preprocessor file not found!")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading preprocessor: {str(e)}")
        return None, None
    
    return models, preprocessor

def predict_income(model, preprocessor, input_data):
    try:
        input_preprocessed = preprocessor.transform(input_data)
        prediction = model.predict(input_preprocessed)
        prediction_proba = model.predict_proba(input_preprocessed)[:, 1]
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None, None

def calculate_metrics(y_true, y_pred, y_pred_proba):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics

with st.spinner("Loading models..."):
    models, preprocessor = load_models()

if models is None or preprocessor is None:
    st.error("❌ Failed to load models. Please check the models directory.")
    st.stop()

st.success("✅ Models loaded successfully!")

expected_features = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]

st.sidebar.header("📊 Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a model",
    list(models.keys())
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tip**: Random Forest generally performs best on this dataset, "
    "but Logistic Regression is more interpretable."
)

st.header("📁 Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=['csv'],
    help="Upload a CSV file with the same features as the training data"
)

use_sample = st.checkbox("Use sample test data (test_data.csv)")

if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

if uploaded_file is not None:
    try:
        test_data_raw = pd.read_csv(uploaded_file)
        test_data = fix_column_names(test_data_raw)
        st.session_state.test_data = test_data
        st.success(f"✅ Data loaded successfully! Shape: {test_data.shape}")
        
        if list(test_data_raw.columns) != list(test_data.columns):
            st.info("🔄 Column names were automatically fixed:")
            for old, new in zip(test_data_raw.columns, test_data.columns):
                if old != new:
                    st.write(f"  • `{old}` → `{new}`")
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.stop()
elif use_sample:
    try:
        test_data_raw = pd.read_csv('test_data.csv')
        test_data = fix_column_names(test_data_raw)
        st.session_state.test_data = test_data
        st.success(f"✅ Sample data loaded successfully! Shape: {test_data.shape}")
    except FileNotFoundError:
        st.error("❌ Sample data file (test_data.csv) not found!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading sample data: {str(e)}")
        st.stop()
else:
    st.info("📤 Please upload a CSV file or use the sample data.")
    st.stop()

with st.expander("📊 View Data Preview"):
    st.dataframe(test_data.head())
    st.write(f"**Total rows:** {len(test_data)}")
    st.write(f"**Total columns:** {len(test_data.columns)}")
    st.write("**Column names:**", ", ".join(test_data.columns))

st.header("🔍 Predict Income")

try:
    X_test_prepared = prepare_data(test_data, expected_features)
    st.success(f"✅ Data prepared successfully! Shape: {X_test_prepared.shape}")
    
    if 'income' in test_data.columns:
        y_test = test_data['income']
        st.info(f"✅ Target column 'income' found. Will evaluate model performance.")
    else:
        y_test = None
        st.info("ℹ️ No target column found. Will make predictions only.")
    
    st.subheader("Features Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Number of features:** {X_test_prepared.shape[1]}")
    with col2:
        st.write(f"**Feature names:** {', '.join(X_test_prepared.columns[:5])}...")
    
except Exception as e:
    st.error(f"❌ Error preparing data: {str(e)}")
    st.stop()

if st.button("🚀 Make Predictions"):
    try:
        model = models[selected_model]
        
        X_test_clean = X_test_prepared.copy()
        
        for col in X_test_clean.columns:
            X_test_clean[col] = pd.to_numeric(X_test_clean[col], errors='coerce')
        
        X_test_clean = X_test_clean.fillna(0)
        
        with st.expander("🔧 Data Types After Cleaning"):
            st.write(X_test_clean.dtypes)
            st.write("First 2 rows of cleaned data:")
            st.dataframe(X_test_clean.head(2))
        
        y_pred, y_pred_proba = predict_income(model, preprocessor, X_test_clean)
        
        if y_pred is not None:
            st.session_state.predictions = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            st.success(f"✅ Predictions made successfully for {len(y_pred)} instances!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", len(y_pred))
            with col2:
                income_high = np.sum(y_pred == 1)
                st.metric("Predicted >50K", income_high)
            with col3:
                income_low = np.sum(y_pred == 0)
                st.metric("Predicted <=50K", income_low)
            
            st.subheader("Prediction Results")
            results_df = pd.DataFrame({
                'Instance': range(1, len(y_pred) + 1),
                'Prediction': ['>50K' if pred == 1 else '<=50K' for pred in y_pred],
                'Probability (>50K)': y_pred_proba
            })
            st.dataframe(results_df)
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
            
            if y_test is not None:
                st.header("📊 Model Performance Evaluation")
                
                metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                    st.metric("Precision", f"{metrics['Precision']:.4f}")
                with col2:
                    st.metric("Recall", f"{metrics['Recall']:.4f}")
                    st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                with col3:
                    st.metric("MCC", f"{metrics['MCC']:.4f}")
                    if 'AUC' in metrics:
                        st.metric("AUC", f"{metrics['AUC']:.4f}")
                
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['<=50K', '>50K'],
                            yticklabels=['<=50K', '>50K'])
                ax.set_title(f'Confusion Matrix - {selected_model}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
                plt.close()
                
                if y_pred_proba is not None:
                    st.subheader("ROC Curve")
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    auc_score = metrics.get('AUC', 0)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, label=f'{selected_model} (AUC = {auc_score:.3f})')
                    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'ROC Curve - {selected_model}')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
    except Exception as e:
        st.error(f"❌ Error making predictions: {str(e)}")
        st.write("🔍 Debug information:")
        if 'X_test_clean' in locals():
            st.write(f"Data shape: {X_test_clean.shape}")
            st.write(f"Data columns: {X_test_clean.columns.tolist()}")
            st.write(f"Data types:\n{X_test_clean.dtypes}")

if y_test is not None and st.session_state.predictions is not None:
    st.header("📊 Model Comparison")
    
    if st.button("Compare All Models"):
        all_metrics = []
        
        with st.spinner("Comparing all models..."):
            for model_name, model in models.items():
                try:
                    y_pred_all, y_pred_proba_all = predict_income(model, preprocessor, X_test_clean)
                    
                    if y_pred_all is not None:
                        metrics_all = calculate_metrics(y_test, y_pred_all, y_pred_proba_all)
                        metrics_all['Model'] = model_name
                        all_metrics.append(metrics_all)
                except Exception as e:
                    st.warning(f"Could not evaluate {model_name}: {str(e)}")
                    continue
            
            if all_metrics:
                comparison_df = pd.DataFrame(all_metrics)
                comparison_df = comparison_df.round(4)
                
                st.subheader("Comparison Table")
                st.dataframe(comparison_df)
                
                st.subheader("Visual Comparison")
                
                metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC']
                if 'AUC' in comparison_df.columns:
                    metrics_to_plot.append('AUC')
                
                fig, ax = plt.subplots(figsize=(12, 6))
                comparison_df_melted = comparison_df.melt(id_vars=['Model'], 
                                                          value_vars=metrics_to_plot,
                                                          var_name='Metric', 
                                                          value_name='Score')
                
                sns.barplot(data=comparison_df_melted, x='Model', y='Score', hue='Metric', ax=ax)
                ax.set_title('Model Performance Comparison')
                ax.set_ylim(0, 1)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
                st.success(f"🏆 **Best performing model**: {best_model['Model']} with Accuracy = {best_model['Accuracy']:.4f}")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>ML Assignment 2 - Income Classification</p>
        <p>Built with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("ℹ️ App Information"):
    st.write(f"**Selected Model**: {selected_model}")
    st.write(f"**Models Loaded**: {list(models.keys())}")
    if 'X_test_clean' in locals():
        st.write(f"**Data Shape**: {X_test_clean.shape}")
    else:
        st.write(f"**Data Shape**: No data loaded")
    st.write(f"**Expected Features**: {len(expected_features)} features")