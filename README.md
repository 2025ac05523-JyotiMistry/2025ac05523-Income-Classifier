# 2025ac05523-Income-Classifier

ML Assignment 2 - Income Classification using 5 ML models with Streamlit
deployment.

## a. Problem Statement

This project aims to classify whether an individual's annual income
exceeds \$50,000 based on demographic and employment-related features
from the Adult Census Income dataset. The task is a binary
classification problem where the target variable is `income` (`<=50K` or
`>50K`).

## b. Dataset Description

-   **Source:** [Adult Census Income](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)
-   **Original Dataset:** 48,842 instances with 15 columns
-   **After Preprocessing:** 30,162 instances (removed rows with missing
    values marked as `?`)
-   **Number of Features:** 14
-   **Target Variable:** `income` (binary: `<=50K = 0`, `>50K = 1`)
-   **Class Distribution:**
    -   `<=50K`: 22,654 (75.1%)
    -   `>50K`: 7,508 (24.9%)

### Feature Types

**Numerical:** - age - fnlwgt - education-num - capital-gain -
capital-loss - hours-per-week

**Categorical:** - workclass - education - marital-status - occupation -
relationship - race - sex - native-country

## c. GitHub Repository Link

[GitHub
Repository](https://github.com/2025ac05523-JyotiMistry/2025ac05523-Income-Classifier)

## d. Live Streamlit App

[Open Live Streamlit
App](https://2025ac05523-income-classifier-jd8cys8e5tmkbvgu7te3pn.streamlit.app/)

The deployed application allows users to: - Upload test data in CSV
format - Select a machine learning model - Generate income predictions -
View evaluation metrics - View the confusion matrix - View the ROC curve

## e. Models Used & Evaluation Metrics

The following five classification models were implemented on the same dataset.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.8543 | 0.9136 | 0.7502 | 0.6218 | 0.6800 | 0.5911 |
| Decision Tree | 0.8158 | 0.7537 | 0.6302 | 0.6298 | 0.6300 | 0.5075 |
| KNN | 0.8341 | 0.8672 | 0.6832 | 0.6218 | 0.6511 | 0.5436 |
| Naive Bayes | 0.6010 | 0.8300 | 0.3795 | 0.9487 | 0.5421 | 0.3876 |
| Random Forest | **0.8556** | **0.9114** | **0.7467** | **0.6358** | **0.6868** | **0.5970** |

### Metric Definitions

| Metric | Description |
|:--|:--|
| Accuracy | Percentage of correctly classified instances |
| AUC | Measures the model's ability to distinguish between the two classes |
| Precision | Proportion of predicted positive cases that are actually positive |
| Recall | Proportion of actual positive cases correctly identified |
| F1 Score | Harmonic mean of precision and recall |
| MCC | Balanced measure of classification quality, especially useful for imbalanced data |

## f. Observations on Model Performance

| Model | Observation |
|:--|:--|
| **Logistic Regression** | Strong overall performance with **85.43% accuracy** and an excellent **AUC of 0.9136**. It provides a good balance between precision and recall. |
| **Decision Tree** | Achieves **81.58% accuracy** with an **AUC of 0.7537**. Precision and recall are almost equal, indicating balanced classification performance. |
| **KNN** | Provides **83.41% accuracy** and a good **AUC of 0.8672**. Its performance is reasonable but lower than Logistic Regression and Random Forest. |
| **Naive Bayes** | Has the lowest **accuracy (60.10%)**, but the highest **recall (94.87%)**. Its low precision indicates a high number of false positives. |
| **Random Forest** | Best overall performance with the highest **accuracy (85.56%)**, **F1 score (0.6868)**, and **MCC (0.5970)**. It also achieves an excellent **AUC of 0.9114**. |

### Overall Winner

**Random Forest** 🏆

Random Forest is the overall best-performing model for this dataset because it achieves the highest accuracy, F1 score, and MCC while maintaining a strong AUC.

## g. How to Run the App

1.  Clone the repository:

``` bash
git clone https://github.com/2025ac05523-JyotiMistry/2025ac05523-Income-Classifier.git
cd 2025ac05523-Income-Classifier
```

2.  Install the required dependencies:

``` bash
pip install -r requirements.txt
```

3.  Run the Streamlit application:

``` bash
streamlit run app.py
```

4.  Open the local Streamlit URL displayed in the terminal.

## h. Project Structure

``` text
2025ac05523-Income-Classifier/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── 2025ac05523_ML2.ipynb
│
└── models/
    ├── model_logistic_regression.pkl
    ├── model_decision_tree.pkl
    ├── model_knn.pkl
    ├── model_naive_bayes.pkl
    ├── model_random_forest.pkl
    └── preprocessor.pkl
```
