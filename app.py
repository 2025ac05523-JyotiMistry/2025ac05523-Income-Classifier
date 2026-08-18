import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Page configuration
st.set_page_config(
    page_title="Income Classification App",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">💰 Income Classification App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict whether income exceeds $50K/year using Machine Learning</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select Section", ["📊 Dataset Upload", "🤖 Model Selection & Prediction", "📈 Model Performance"])

# Define column lists (same as in training)
CATEGORICAL_COLS = ['workclass', 'education', 'marital.status', 'occupation', 
                    'relationship', 'race', 'sex', 'native.country']
NUMERICAL_COLS = ['age', 'fnlwgt', 'education.num', 'capital.gain', 
                  'capital.loss', 'hours.per.week']

# Load preprocessor
@st.cache_resource
def load_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), NUMERICAL_COLS),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), CATEGORICAL_COLS)
        ])
    return preprocessor

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'models/Logistic_Regression.pkl',
        'Decision Tree': 'models/Decision_Tree.pkl',
        'KNN': 'models/KNN.pkl',
        'Naive Bayes': 'models/Naive_Bayes.pkl',
        'Random Forest': 'models/Random_Forest.pkl'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
            except Exception as e:
                st.warning(f"Could not load {name}: {str(e)}")
    return models

# Preprocess data function
def preprocess_data(data, preprocessor):
    X = data.copy()
    # Select only the columns we need
    available_cols = [col for col in CATEGORICAL_COLS + NUMERICAL_COLS if col in X.columns]
    X = X[available_cols]
    # Preprocess
    X_processed = preprocessor.fit_transform(X)
    return X_processed, X

# Make predictions
def make_prediction(model, data, preprocessor):
    X_processed, X_clean = preprocess_data(data, preprocessor)
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:, 1] if hasattr(model, 'predict_proba') else None
    return predictions, probabilities, X_clean

# Section 1: Dataset Upload
if option == "📊 Dataset Upload":
    st.header("📊 Upload Test Data")
    st.write("Upload your test CSV file to evaluate the models. The dataset should have the same features as the training data.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            test_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Data uploaded successfully! Shape: {test_data.shape}")
            
            # Store in session state
            st.session_state['test_data'] = test_data
            
            # Display data preview
            st.subheader("📋 Data Preview")
            st.dataframe(test_data.head(10))
            
            # Basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Rows", test_data.shape[0])
            with col2:
                st.metric("Number of Columns", test_data.shape[1])
            with col3:
                missing = test_data.isnull().sum().sum()
                st.metric("Missing Values", missing)
            
            # Show column info
            with st.expander("📊 Column Information"):
                col_info = pd.DataFrame({
                    'Column': test_data.columns,
                    'Data Type': test_data.dtypes.values,
                    'Unique Values': [test_data[col].nunique() for col in test_data.columns],
                    'Missing Values': [test_data[col].isnull().sum() for col in test_data.columns]
                })
                st.dataframe(col_info)
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

# Section 2: Model Selection & Prediction
elif option == "🤖 Model Selection & Prediction":
    st.header("🤖 Model Selection & Prediction")
    
    if 'test_data' not in st.session_state:
        st.warning("⚠️ Please upload test data first in the 'Dataset Upload' section")
    else:
        test_data = st.session_state['test_data']
        
        # Load models
        models = load_models()
        preprocessor = load_preprocessor()
        
        if not models:
            st.error("❌ No models found. Please train models first and save them in the 'models/' folder.")
        else:
            # Model selection
            selected_model_name = st.selectbox("Select Model", list(models.keys()))
            selected_model = models[selected_model_name]
            
            # Display model info
            st.info(f"Selected Model: **{selected_model_name}**")
            
            # Make predictions
            if st.button("🔮 Make Predictions"):
                with st.spinner("Making predictions..."):
                    try:
                        # Predict
                        y_pred, y_pred_proba, X_clean = make_prediction(selected_model, test_data, preprocessor)
                        
                        # Store predictions
                        st.session_state['predictions'] = y_pred
                        st.session_state['probabilities'] = y_pred_proba
                        st.session_state['selected_model'] = selected_model_name
                        
                        # Display predictions summary
                        st.subheader("📊 Prediction Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        pred_counts = pd.Series(y_pred).value_counts()
                        with col1:
                            st.metric("Total Predictions", len(y_pred))
                        with col2:
                            st.metric("Predicted <=50K", pred_counts.get(0, 0))
                        with col3:
                            st.metric("Predicted >50K", pred_counts.get(1, 0))
                        
                        # Show probability distribution
                        if y_pred_proba is not None:
                            fig = px.histogram(
                                x=y_pred_proba,
                                nbins=50,
                                title="Probability Distribution (>50K)",
                                labels={'x': 'Probability', 'y': 'Count'},
                                color_discrete_sequence=['#4CAF50']
                            )
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show predictions in table
                        st.subheader("📋 Prediction Results")
                        result_df = test_data.copy()
                        result_df['Prediction'] = y_pred
                        result_df['Prediction_Label'] = result_df['Prediction'].map({0: '<=50K', 1: '>50K'})
                        if y_pred_proba is not None:
                            result_df['Probability_>50K'] = y_pred_proba
                        
                        # Select columns to display
                        display_cols = ['age', 'occupation', 'education', 'Prediction_Label']
                        if y_pred_proba is not None:
                            display_cols.append('Probability_>50K')
                        available_cols = [col for col in display_cols if col in result_df.columns]
                        
                        st.dataframe(result_df[available_cols].head(20))
                        
                        # Download results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Predictions CSV",
                            data=csv,
                            file_name=f"predictions_{selected_model_name.replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error making predictions: {str(e)}")

# Section 3: Model Performance
else:
    st.header("📈 Model Performance Comparison")
    
    # Results from your notebook
    try:
        results_df = pd.DataFrame([
            {'Model': 'Logistic Regression', 'Accuracy': 0.8543, 'AUC': 0.9136, 'Precision': 0.7502, 'Recall': 0.6218, 'F1': 0.6800, 'MCC': 0.5911},
            {'Model': 'Decision Tree', 'Accuracy': 0.8158, 'AUC': 0.7537, 'Precision': 0.6302, 'Recall': 0.6298, 'F1': 0.6300, 'MCC': 0.5075},
            {'Model': 'KNN', 'Accuracy': 0.8341, 'AUC': 0.8672, 'Precision': 0.6832, 'Recall': 0.6218, 'F1': 0.6511, 'MCC': 0.5436},
            {'Model': 'Naive Bayes', 'Accuracy': 0.6010, 'AUC': 0.8300, 'Precision': 0.3795, 'Recall': 0.9487, 'F1': 0.5421, 'MCC': 0.3876},
            {'Model': 'Random Forest', 'Accuracy': 0.8556, 'AUC': 0.9114, 'Precision': 0.7467, 'Recall': 0.6358, 'F1': 0.6868, 'MCC': 0.5970}
        ])
        
        # Display metrics table
        st.subheader("📊 Model Performance Metrics")
        st.dataframe(
            results_df.style.background_gradient(subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'], cmap='Blues')
        )
        
        # Visualization - Bar Chart
        st.subheader("📊 Performance Comparison Charts")
        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        
        # Bar chart
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df['Model'],
                y=results_df[metric],
                text=results_df[metric].round(3),
                textposition='auto',
            ))
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        heatmap_data = results_df.set_index('Model')[metrics].T
        fig_heatmap = px.imshow(
            heatmap_data, 
            text_auto=True, 
            color_continuous_scale='Blues',
            title="Performance Heatmap"
        )
        fig_heatmap.update_layout(height=450)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Confusion Matrix (if actual labels available)
        if 'test_data' in st.session_state and 'predictions' in st.session_state:
            st.subheader("📊 Confusion Matrix")
            
            test_data = st.session_state['test_data']
            if 'income' in test_data.columns:
                y_true = test_data['income'].map({'<=50K': 0, '>50K': 1})
                y_pred = st.session_state['predictions']
                
                cm = confusion_matrix(y_true, y_pred)
                
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    x=['Predicted <=50K', 'Predicted >50K'],
                    y=['Actual <=50K', 'Actual >50K'],
                    color_continuous_scale='Blues',
                    title="Confusion Matrix"
                )
                fig_cm.update_layout(height=400)
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Classification Report
                with st.expander("📋 Classification Report"):
                    report = classification_report(y_true, y_pred, target_names=['<=50K', '>50K'])
                    st.text(report)
            else:
                st.info("💡 Upload a dataset with 'income' column to see confusion matrix.")
        
        # Observations
        st.subheader("📝 Model Performance Observations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🏆 Best Model: Random Forest**
            - Highest Accuracy: 85.56%
            - Highest AUC: 91.14%
            - Best F1 Score: 68.68%
            - Best MCC: 59.70%
            
            **Why Random Forest performs best:**
            - Handles non-linear relationships well
            - Robust to overfitting
            - Handles both numerical and categorical features effectively
            - Ensemble method captures complex patterns
            """)
        
        with col2:
            st.markdown("""
            **📊 Model Observations:**
            - **Logistic Regression**: Strong performer, excellent AUC (91.36%), most interpretable
            - **Decision Tree**: Good but lowest AUC, prone to overfitting
            - **KNN**: Reasonable performance but sensitive to scaling
            - **Naive Bayes**: Highest recall (94.87%) but poor precision (37.95%)
            - **Random Forest**: Best overall performance
            
            **Overall Winner: Random Forest** 🏆
            """)
            
    except Exception as e:
        st.warning(f"⚠️ Could not load performance data. Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for ML Assignment 2 - BITS Pilani")