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

-   **Source:** [Kaggle - Adult Census Income](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data)
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

  -------------------------------------------------------------------------------
  ML Model       Accuracy        AUC   Precision     Recall   F1 Score        MCC
  Name                                                                 
  ------------ ---------- ---------- ----------- ---------- ---------- ----------
  Logistic         0.8543     0.9136      0.7502     0.6218     0.6800     0.5911
  Regression                                                           

  Decision         0.8158     0.7537      0.6302     0.6298     0.6300     0.5075
  Tree                                                                 

  K-Nearest        0.8341     0.8672      0.6832     0.6218     0.6511     0.5436
  Neighbors                                                            

  Naive Bayes      0.6010     0.8300      0.3795     0.9487     0.5421     0.3876

  Random           0.8556     0.9114      0.7467     0.6358     0.6868     0.5970
  Forest                                                               
  -------------------------------------------------------------------------------

## f. Observations on Model Performance

  -----------------------------------------------------------------------
  ML Model Name                       Observation about Model Performance
  ----------------------------------- -----------------------------------
  Logistic Regression                 Strong overall performer with
                                      85.43% accuracy and excellent AUC
                                      of 0.9136. Shows a good balance
                                      between precision (75.02%) and
                                      recall (62.18%). MCC of 0.5911
                                      indicates substantial correlation.

  Decision Tree                       Moderate performance with 81.58%
                                      accuracy and the lowest AUC among
                                      the models (0.7537). Shows balanced
                                      precision (63.02%) and recall
                                      (62.98%).

  K-Nearest Neighbors                 Reasonable performance with 83.41%
                                      accuracy and strong AUC of 0.8672.
                                      Precision is 68.32% and recall is
                                      62.18%. Performance is sensitive to
                                      feature scaling.

  Naive Bayes                         Lowest overall accuracy at 60.10%.
                                      It has the highest recall (94.87%)
                                      but low precision (37.95%),
                                      resulting in many false positives.

  Random Forest                       Best overall performer with the
                                      highest accuracy (85.56%), F1 score
                                      (0.6868), and MCC (0.5970). It also
                                      provides an excellent AUC of
                                      0.9114.
  -----------------------------------------------------------------------

### Overall Winner for the Dataset: Random Forest 🏆

Random Forest achieves the highest accuracy (85.56%), highest F1 score
(0.6868), and highest MCC (0.5970), with an excellent AUC of 0.9114. The
ensemble approach captures non-linear relationships and performs well on
the dataset.

### Key Insights

-   **Best Overall:** Random Forest
-   **Best for Interpretability:** Logistic Regression
-   **Best for Recall:** Naive Bayes
-   **Worst Performing by Accuracy:** Naive Bayes

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
