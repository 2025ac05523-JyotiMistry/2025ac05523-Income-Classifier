# 2025ac05523-Income-Classifier
ML Assignment 2 - Income Classification using 5 ML models with Streamlit deployment.

## a. Problem Statement
This project aims to classify whether an individual's annual income exceeds $50,000 based on demographic and employment-related features from the Adult Census Income dataset. The classification helps in understanding socio-economic patterns and can be used for targeted policy-making or business decisions. The task is a binary classification problem where the target variable is "income" (<=50K or >50K).

## b. Dataset Description
- **Source**: UCI Adult Census Income Dataset (https://archive.ics.uci.edu/ml/datasets/adult)
- **Number of Instances**: 30,162 (after removing missing values)
- **Number of Features**: 14
- **Target Variable**: income (binary: <=50K = 0, >50K = 1)
- **Class Distribution**: 
  - <=50K: 22,654 (75.1%)
  - >50K: 7,508 (24.9%)
- **Feature Types**:
  - Numerical: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country

## c. GitHub Repository Link
[https://github.com/2025ac05523-JyotiMistry/2025ac05523-Income-Classifier.git](https://github.com/2025ac05523-JyotiMistry/2025ac05523-Income-Classifier.git)

## d. Models Used & Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8543 | 0.9136 | 0.7502 | 0.6218 | 0.6800 | 0.5911 |
| Decision Tree | 0.8158 | 0.7537 | 0.6302 | 0.6298 | 0.6300 | 0.5075 |
| K-Nearest Neighbors | 0.8341 | 0.8672 | 0.6832 | 0.6218 | 0.6511 | 0.5436 |
| Naive Bayes | 0.6010 | 0.8300 | 0.3795 | 0.9487 | 0.5421 | 0.3876 |
| Random Forest | 0.8556 | 0.9114 | 0.7467 | 0.6358 | 0.6868 | 0.5970 |

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | **Strong overall performer** with 85.43% accuracy and excellent AUC of 0.9136. Shows good balance between precision (75.02%) and recall (62.18%). MCC of 0.5911 indicates substantial correlation. Linear model works well due to linearly separable patterns in the data. |
| Decision Tree | Moderate performance with 81.58% accuracy. Lowest AUC among all models (0.7537), suggesting poor ranking capability. Shows balanced precision (63.02%) and recall (62.98%). MCC of 0.5075 indicates moderate correlation. Struggles with generalization and is susceptible to overfitting despite being a tree-based model. |
| K-Nearest Neighbors | Reasonable performance with 83.41% accuracy and strong AUC of 0.8672. Precision is decent at 68.32% but recall is lower at 62.18%. MCC of 0.5436 shows moderate correlation. Lazy learning approach works moderately well but computational cost increases with test data size. |
| Naive Bayes | **Lowest overall accuracy** at just 60.10%. Has **extremely high recall of 94.87%** (best among all models) but **very poor precision of 37.95%**, indicating it predicts the positive class too aggressively with many false positives. AUC of 0.8300 is reasonable. Lowest MCC of 0.3876 shows weak correlation. The independence assumption fails for this dataset. |
| Random Forest | **Best overall performer** with highest accuracy (85.56%), excellent AUC (0.9114), best F1 score (0.6868), and highest MCC (0.5970). Shows strong precision (74.67%) and recall (63.58%). Ensemble method effectively captures complex patterns and generalizes well without overfitting. |

### Overall Winner for your dataset: **Random Forest**

**Justification:** Random Forest achieves the highest accuracy (85.56%), highest F1 score (0.6868), highest MCC (0.5970), and excellent AUC (0.9114). The ensemble approach of combining multiple decision trees successfully handles the imbalanced nature of the dataset (75% <=50K, 25% >50K) and captures non-linear relationships between features and income. Logistic Regression is a close second with slightly lower metrics, making it a simpler alternative if interpretability is prioritized over marginal performance gains.

## f. How to Run the App
1. Clone the repository# 2025ac05523-Income-Classifier
ML Assignment 2 - Income Classification using 5 ML models with Streamlit deployment.

## a. Problem Statement
This project aims to classify whether an individual's annual income exceeds $50,000 based on demographic and employment-related features from the Adult Census Income dataset. The classification helps in understanding socio-economic patterns and can be used for targeted policy-making or business decisions. The task is a binary classification problem where the target variable is "income" (<=50K or >50K).

## b. Dataset Description
- **Source**: UCI Adult Census Income Dataset (https://archive.ics.uci.edu/ml/datasets/adult)
- **Number of Instances**: 30,162 (after removing missing values)
- **Number of Features**: 14
- **Target Variable**: income (binary: <=50K = 0, >50K = 1)
- **Class Distribution**: 
  - <=50K: 22,654 (75.1%)
  - >50K: 7,508 (24.9%)
- **Feature Types**:
  - Numerical: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country

## c. GitHub Repository Link
[https://github.com/2025ac05523-JyotiMistry/2025ac05523-Income-Classifier.git](https://github.com/2025ac05523-JyotiMistry/2025ac05523-Income-Classifier.git)

## d. Models Used & Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8543 | 0.9136 | 0.7502 | 0.6218 | 0.6800 | 0.5911 |
| Decision Tree | 0.8158 | 0.7537 | 0.6302 | 0.6298 | 0.6300 | 0.5075 |
| K-Nearest Neighbors | 0.8341 | 0.8672 | 0.6832 | 0.6218 | 0.6511 | 0.5436 |
| Naive Bayes | 0.6010 | 0.8300 | 0.3795 | 0.9487 | 0.5421 | 0.3876 |
| Random Forest | 0.8556 | 0.9114 | 0.7467 | 0.6358 | 0.6868 | 0.5970 |

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | **Strong overall performer** with 85.43% accuracy and excellent AUC of 0.9136. Shows good balance between precision (75.02%) and recall (62.18%). MCC of 0.5911 indicates substantial correlation. Linear model works well due to linearly separable patterns in the data. Best performing linear model. |
| Decision Tree | Moderate performance with 81.58% accuracy. **Lowest AUC among all models (0.7537)**, suggesting poor ranking capability. Shows balanced precision (63.02%) and recall (62.98%). MCC of 0.5075 indicates moderate correlation. Struggles with generalization and is susceptible to overfitting despite being a tree-based model. |
| K-Nearest Neighbors | Reasonable performance with 83.41% accuracy and strong AUC of 0.8672. Precision is decent at 68.32% but recall is lower at 62.18%. MCC of 0.5436 shows moderate correlation. Lazy learning approach works moderately well but computational cost increases with test data size. Sensitive to feature scaling. |
| Naive Bayes | **Lowest overall accuracy** at just 60.10%. Has **extremely high recall of 94.87%** (best among all models) but **very poor precision of 37.95%**, indicating it predicts the positive class too aggressively with many false positives. AUC of 0.8300 is reasonable. **Lowest MCC of 0.3876** shows weak correlation. The independence assumption fails for this dataset. Best at identifying positive instances but at the cost of many false alarms. |
| Random Forest | **Best overall performer** with highest accuracy (85.56%), excellent AUC (0.9114), **best F1 score (0.6868)**, and **highest MCC (0.5970)**. Shows strong precision (74.67%) and recall (63.58%). Ensemble method effectively captures complex patterns and generalizes well without overfitting. Handles both numerical and categorical features effectively. |

### Overall Winner for your dataset: **Random Forest** 🏆

**Justification:** Random Forest achieves the highest accuracy (85.56%), highest F1 score (0.6868), highest MCC (0.5970), and excellent AUC (0.9114). The ensemble approach of combining multiple decision trees successfully handles the imbalanced nature of the dataset (75% <=50K, 25% >50K) and captures non-linear relationships between features and income. Logistic Regression is a close second with slightly lower metrics, making it a simpler alternative if interpretability is prioritized over marginal performance gains.

**Key Insights:**
- **Best Overall**: Random Forest
- **Best for Interpretability**: Logistic Regression
- **Best for Identifying High-Income Individuals (Recall)**: Naive Bayes (but watch out for false positives!)
- **Worst Performing**: Naive Bayes (overall accuracy)

## f. How to Run the App
1. Clone the repository::