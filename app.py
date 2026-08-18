import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Income Classifier",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("💰 Income Classifier")

st.markdown(
    """
    This application predicts whether an individual's income
    exceeds $50,000 using five machine learning classification models.
    """
)


# ============================================================
# EXPECTED INPUT COLUMNS
# ============================================================

EXPECTED_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country"
]


# ============================================================
# LOAD MODELS AND PREPROCESSOR
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    model_files = {
        "Logistic Regression":
            "models/model_logistic_regression.pkl",

        "Decision Tree":
            "models/model_decision_tree.pkl",

        "KNN":
            "models/model_knn.pkl",

        "Naive Bayes":
            "models/model_naive_bayes.pkl",

        "Random Forest":
            "models/model_random_forest.pkl"
    }

    # --------------------------------------------------------
    # Load ML models
    # --------------------------------------------------------

    for model_name, model_path in model_files.items():

        try:

            with open(model_path, "rb") as file:
                models[model_name] = pickle.load(file)

        except FileNotFoundError:

            st.error(
                f"❌ Model file not found: {model_path}"
            )

            return None, None, None

        except Exception as e:

            st.error(
                f"❌ Error loading {model_name}: {str(e)}"
            )

            return None, None, None


    # --------------------------------------------------------
    # Load preprocessor
    # --------------------------------------------------------

    try:

        with open(
            "models/preprocessor.pkl",
            "rb"
        ) as file:

            preprocessor_data = pickle.load(file)


        # IMPORTANT:
        # Your preprocessor.pkl was saved as a dictionary.
        # The actual transformer is inside:
        #
        # preprocessor_data["preprocessor"]

        if isinstance(preprocessor_data, dict):

            preprocessor = preprocessor_data["preprocessor"]

            saved_columns = preprocessor_data.get(
                "expected_columns",
                EXPECTED_COLUMNS
            )

        else:

            # Fallback in case preprocessor.pkl contains
            # the transformer directly.

            preprocessor = preprocessor_data

            saved_columns = EXPECTED_COLUMNS


    except FileNotFoundError:

        st.error(
            "❌ models/preprocessor.pkl was not found."
        )

        return None, None, None

    except Exception as e:

        st.error(
            f"❌ Error loading preprocessor: {str(e)}"
        )

        return None, None, None


    return models, preprocessor, saved_columns


# ============================================================
# LOAD EVERYTHING
# ============================================================

models, preprocessor, saved_columns = load_models()


if models is None or preprocessor is None:

    st.error(
        "❌ Failed to load models or preprocessor. "
        "Please check the models directory."
    )

    st.stop()


# ============================================================
# SIDEBAR - MODEL SELECTION
# ============================================================

st.sidebar.header("🤖 Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a model",
    list(models.keys())
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"**Selected Model:** {selected_model}"
)


# ============================================================
# UPLOAD TEST DATA
# ============================================================

st.header("📂 Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        test_data = pd.read_csv(uploaded_file)


        st.success(
            f"✅ Data loaded: "
            f"{test_data.shape[0]} rows, "
            f"{test_data.shape[1]} columns"
        )


        # ----------------------------------------------------
        # Data Preview
        # ----------------------------------------------------

        st.subheader("📊 Data Preview")

        st.dataframe(
            test_data.head(),
            use_container_width=True
        )


        # ----------------------------------------------------
        # Show columns
        # ----------------------------------------------------

        with st.expander("🔍 View CSV Columns"):

            st.write(
                list(test_data.columns)
            )


        # ====================================================
        # MAKE PREDICTIONS BUTTON
        # ====================================================

        if st.button(
            "🚀 Make Predictions",
            type="primary"
        ):

            try:

                # ------------------------------------------------
                # Copy input data
                # ------------------------------------------------

                X_test = test_data.copy()


                # ------------------------------------------------
                # Separate target column
                # ------------------------------------------------

                y_test = None

                if "income" in X_test.columns:

                    y_test = X_test["income"].copy()

                    X_test = X_test.drop(
                        "income",
                        axis=1
                    )


                # ------------------------------------------------
                # Check required columns
                # ------------------------------------------------

                missing_columns = [
                    column
                    for column in EXPECTED_COLUMNS
                    if column not in X_test.columns
                ]


                if missing_columns:

                    st.error(
                        "❌ Missing required columns:"
                    )

                    st.write(
                        missing_columns
                    )

                    st.stop()


                # ------------------------------------------------
                # Remove unexpected columns
                # ------------------------------------------------

                X_test = X_test[
                    EXPECTED_COLUMNS
                ]


                # ------------------------------------------------
                # Get selected model
                # ------------------------------------------------

                model = models[selected_model]


                # ------------------------------------------------
                # PREPROCESS TEST DATA
                #
                # THIS IS THE IMPORTANT FIX
                # ------------------------------------------------

                X_test_processed = (
                    preprocessor.transform(X_test)
                )


                # ------------------------------------------------
                # Make predictions
                # ------------------------------------------------

                y_pred = model.predict(
                    X_test_processed
                )


                # ------------------------------------------------
                # Prediction probabilities
                # ------------------------------------------------

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    y_pred_proba = (
                        model.predict_proba(
                            X_test_processed
                        )[:, 1]
                    )

                else:

                    y_pred_proba = None


                # =================================================
                # PREDICTION RESULTS
                # =================================================

                st.header("🎯 Predictions")


                # Convert prediction values
                # into readable income labels

                prediction_labels = []

                for prediction in y_pred:

                    if prediction == 1:

                        prediction_labels.append(
                            ">50K"
                        )

                    else:

                        prediction_labels.append(
                            "<=50K"
                        )


                # Probability column

                if y_pred_proba is not None:

                    probability_values = [
                        f"{value:.4f}"
                        for value in y_pred_proba
                    ]

                else:

                    probability_values = [
                        "N/A"
                        for _ in y_pred
                    ]


                results = pd.DataFrame(
                    {
                        "Prediction":
                            prediction_labels,

                        "Probability (>50K)":
                            probability_values
                    }
                )


                st.dataframe(
                    results,
                    use_container_width=True
                )


                # ------------------------------------------------
                # Download predictions
                # ------------------------------------------------

                prediction_csv = (
                    results.to_csv(
                        index=False
                    )
                )


                st.download_button(
                    label="📥 Download Predictions",
                    data=prediction_csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )


                # =================================================
                # MODEL EVALUATION
                # =================================================

                if y_test is not None:

                    st.header(
                        "📈 Model Evaluation"
                    )


                    # ------------------------------------------------
                    # Convert target values if necessary
                    # ------------------------------------------------

                    y_true = y_test.copy()


                    # Handle string labels such as:
                    #
                    # <=50K
                    # >50K
                    #
                    # and convert them to 0 / 1.

                    if y_true.dtype == "object":

                        y_true = (
                            y_true
                            .astype(str)
                            .str.strip()
                            .replace(
                                {
                                    "<=50K": 0,
                                    ">50K": 1,
                                    "<=50K.": 0,
                                    ">50K.": 1
                                }
                            )
                        )


                    # Convert to numeric if possible

                    try:

                        y_true = pd.to_numeric(
                            y_true
                        )

                    except Exception:

                        pass


                    y_true = np.array(
                        y_true
                    )


                    y_pred_numeric = np.array(
                        y_pred
                    )


                    # ------------------------------------------------
                    # Calculate metrics
                    # ------------------------------------------------

                    accuracy = accuracy_score(
                        y_true,
                        y_pred_numeric
                    )


                    precision = precision_score(
                        y_true,
                        y_pred_numeric,
                        zero_division=0
                    )


                    recall = recall_score(
                        y_true,
                        y_pred_numeric,
                        zero_division=0
                    )


                    f1 = f1_score(
                        y_true,
                        y_pred_numeric,
                        zero_division=0
                    )


                    mcc = matthews_corrcoef(
                        y_true,
                        y_pred_numeric
                    )


                    # AUC

                    if y_pred_proba is not None:

                        auc = roc_auc_score(
                            y_true,
                            y_pred_proba
                        )

                    else:

                        auc = 0.0


                    metrics = {
                        "Accuracy": accuracy,
                        "AUC": auc,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1,
                        "MCC": mcc
                    }


                    # =================================================
                    # DISPLAY METRICS
                    # =================================================

                    metric_columns = st.columns(
                        3
                    )


                    for index, (
                        metric_name,
                        metric_value
                    ) in enumerate(
                        metrics.items()
                    ):

                        metric_columns[
                            index % 3
                        ].metric(
                            metric_name,
                            f"{metric_value:.4f}"
                        )


                    # =================================================
                    # CONFUSION MATRIX
                    # =================================================

                    st.subheader(
                        "📊 Confusion Matrix"
                    )


                    cm = confusion_matrix(
                        y_true,
                        y_pred_numeric
                    )


                    fig_cm, ax_cm = plt.subplots(
                        figsize=(6, 4)
                    )


                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        ax=ax_cm,
                        xticklabels=[
                            "<=50K",
                            ">50K"
                        ],
                        yticklabels=[
                            "<=50K",
                            ">50K"
                        ]
                    )


                    ax_cm.set_xlabel(
                        "Predicted"
                    )

                    ax_cm.set_ylabel(
                        "Actual"
                    )

                    ax_cm.set_title(
                        f"{selected_model} - Confusion Matrix"
                    )


                    st.pyplot(
                        fig_cm
                    )

                    plt.close(
                        fig_cm
                    )


                    # =================================================
                    # ROC CURVE
                    # =================================================

                    if y_pred_proba is not None:

                        st.subheader(
                            "📈 ROC Curve"
                        )


                        fpr, tpr, thresholds = (
                            roc_curve(
                                y_true,
                                y_pred_proba
                            )
                        )


                        fig_roc, ax_roc = (
                            plt.subplots(
                                figsize=(6, 4)
                            )
                        )


                        ax_roc.plot(
                            fpr,
                            tpr,
                            label=(
                                f"{selected_model} "
                                f"(AUC = {auc:.3f})"
                            )
                        )


                        ax_roc.plot(
                            [0, 1],
                            [0, 1],
                            "k--"
                        )


                        ax_roc.set_xlabel(
                            "False Positive Rate"
                        )

                        ax_roc.set_ylabel(
                            "True Positive Rate"
                        )

                        ax_roc.set_title(
                            f"{selected_model} - ROC Curve"
                        )


                        ax_roc.legend()


                        st.pyplot(
                            fig_roc
                        )


                        plt.close(
                            fig_roc
                        )


            # ========================================================
            # PREDICTION ERROR
            # ========================================================

            except Exception as e:

                st.error(
                    f"❌ Prediction error: {str(e)}"
                )

                st.info(
                    """
                    Please ensure your CSV contains these columns:

                    age, workclass, fnlwgt, education,
                    education-num, marital-status, occupation,
                    relationship, race, sex, capital-gain,
                    capital-loss, hours-per-week, native-country

                    The optional target column is:
                    income
                    """
                )


    # ============================================================
    # CSV READING ERROR
    # ============================================================

    except Exception as e:

        st.error(
            f"❌ Error reading CSV file: {str(e)}"
        )


# ================================================================
# FOOTER
# ================================================================

st.markdown("---")

st.caption(
    "ML Assignment 2 - Income Classification"
)