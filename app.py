# Income Classification App
# Streamlit Application for ML Assignment 2

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

# Set page configuration
st.set_page_config(
    page_title="Income Classifier",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
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

# 1. Title and Description
# App title and description
st.markdown('<p class="main-header">💰 Income Classifier</p>', unsafe_allow_html=True)

st.markdown("""
This application uses machine learning models to predict whether an individual's 
annual income exceeds $50,000 based on demographic and employment-related features.

**Dataset**: Adult Census Income Dataset (UCI)
""")

# 2. Helper Functions

# Function to load models
@st.cache_resource
def load_models():
    """
    Load all trained models and preprocessor from .pkl files
    """
    models = {}
    model_paths = {
        'Logistic Regression': 'models/model_logistic_regression.pkl',
        'Decision Tree': 'models/model_decision_tree.pkl',
        'KNN': 'models/model_knn.pkl',
        'Naive Bayes': 'models/model_naive_bayes.pkl',
        'Random Forest': 'models/model_random_forest.pkl'
    }
    
    # Check if models directory exists
    if not os.path.exists('models'):
        st.error("❌ Models directory not found! Please ensure the models folder exists.")
        return None, None
    
    # Load each model
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
    
    # Load preprocessor
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

# Function to make predictions
def predict_income(model, preprocessor, input_data):
    """
    Make predictions using the selected model
    """
    try:
        # Preprocess the input data
        input_preprocessed = preprocessor.transform(input_data)
        
        # Make predictions
        prediction = model.predict(input_preprocessed)
        prediction_proba = model.predict_proba(input_preprocessed)[:, 1]
        
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None, None

# Function to calculate metrics
def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate all evaluation metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    # Calculate AUC if probability scores are available
    if y_pred_proba is not None:
        metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics

# 3. Load Models

# Load models and preprocessor
with st.spinner("Loading models..."):
    models, preprocessor = load_models()

if models is None or preprocessor is None:
    st.error("❌ Failed to load models. Please check the models directory.")
    st.stop()

st.success("✅ Models loaded successfully!")

# Display available models
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

# 4. Data Upload Section
st.header("📁 Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=['csv'],
    help="Upload a CSV file with the same features as the training data"
)

# Option to use sample data
use_sample = st.checkbox("Use sample test data (test_data.csv)")

# 5. Process Uploaded Data
# Initialize session state for data
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Load data
if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)
        st.session_state.test_data = test_data
        st.success(f"✅ Data loaded successfully! Shape: {test_data.shape}")
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.stop()
elif use_sample:
    try:
        test_data = pd.read_csv('test_data.csv')
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

# Display data preview
with st.expander("📊 View Data Preview"):
    st.dataframe(test_data.head())
    st.write(f"**Total rows**: {len(test_data)}")
    st.write(f"**Total columns**: {len(test_data.columns)}")

# 6. Feature Selection for Prediction
st.header("🔍 Predict Income")

# Separate features and target (if available)
if 'income' in test_data.columns:
    X_test = test_data.drop('income', axis=1)
    y_test = test_data['income']
    st.info(f"Target column 'income' found. Will evaluate model performance.")
else:
    X_test = test_data
    y_test = None
    st.info("No target column found. Will make predictions only.")

# Display feature info
st.subheader("Features Summary")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Number of features**: {X_test.shape[1]}")
with col2:
    st.write(f"**Feature names**: {', '.join(X_test.columns[:5])}...")

# 7. Make Predictions
# Prediction button
if st.button("🚀 Make Predictions"):
    try:
        # Get the selected model
        model = models[selected_model]
        
        # Make predictions
        y_pred, y_pred_proba = predict_income(model, preprocessor, X_test)
        
        if y_pred is not None:
            st.session_state.predictions = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Display prediction results
            st.success(f"✅ Predictions made successfully for {len(y_pred)} instances!")
            
            # Show prediction summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", len(y_pred))
            with col2:
                income_high = np.sum(y_pred == 1)
                st.metric("Predicted >50K", income_high)
            with col3:
                income_low = np.sum(y_pred == 0)
                st.metric("Predicted <=50K", income_low)
            
            # Display prediction results
            st.subheader("Prediction Results")
            results_df = pd.DataFrame({
                'Instance': range(1, len(y_pred) + 1),
                'Prediction': ['>50K' if pred == 1 else '<=50K' for pred in y_pred],
                'Probability (>50K)': y_pred_proba
            })
            st.dataframe(results_df)
            
            # Download predictions
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"❌ Error making predictions: {str(e)}")

# 8. Model Evaluation 
# Evaluate model performance if ground truth is available
if y_test is not None and st.session_state.predictions is not None:
    st.header("📊 Model Performance Evaluation")
    
    y_pred = st.session_state.predictions['y_pred']
    y_pred_proba = st.session_state.predictions['y_pred_proba']
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Display metrics in columns
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
    
    # Display comparison with table
    st.subheader("📋 Full Results")
    
    # Create results table
    results_table = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    results_table = results_table.round(4)
    st.table(results_table)
    
    # 9. Visualizations
    # Confusion Matrix
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
    
    # ROC Curve
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

# Display comparison with all models if ground truth is available
if y_test is not None and st.session_state.predictions is not None:
    st.header("📊 Model Comparison")
    
    if st.button("Compare All Models"):
        all_metrics = []
        
        for model_name, model in models.items():
            try:
                # Make predictions
                y_pred_all, y_pred_proba_all = predict_income(model, preprocessor, X_test)
                
                if y_pred_all is not None:
                    # Calculate metrics
                    metrics_all = calculate_metrics(y_test, y_pred_all, y_pred_proba_all)
                    metrics_all['Model'] = model_name
                    all_metrics.append(metrics_all)
            except:
                continue
        
        if all_metrics:
            # Create comparison dataframe
            comparison_df = pd.DataFrame(all_metrics)
            comparison_df = comparison_df.round(4)
            
            # Display comparison table
            st.subheader("Comparison Table")
            st.dataframe(comparison_df)
            
            # Visualize comparison
            st.subheader("Visual Comparison")
            
            # Select metrics to plot
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC']
            if 'AUC' in comparison_df.columns:
                metrics_to_plot.append('AUC')
            
            # Create bar chart
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
            
            # Identify best model
            best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
            st.success(f"🏆 **Best performing model**: {best_model['Model']} with Accuracy = {best_model['Accuracy']:.4f}")

# 11. Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>ML Assignment 2 - Income Classification</p>
        <p>using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Display session state info for debugging
with st.expander("ℹ️ App Information"):
    st.write(f"**Selected Model**: {selected_model}")
    st.write(f"**Models Loaded**: {list(models.keys())}")
    st.write(f"**Data Shape**: {X_test.shape if X_test is not None else 'No data loaded'}")