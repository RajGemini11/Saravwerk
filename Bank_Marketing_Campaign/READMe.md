# Bank Marketing Campaign Analysis: Comparing Classifiers

## Project Overview

This project analyzes a Portuguese banking institution's telemarketing campaign data to predict whether a client will subscribe to a term deposit. The analysis compares the performance of four machine learning classifiers: K-Nearest Neighbors (KNN), Logistic Regression, Decision Trees, and Support Vector Machines (SVM).

## Dataset Information

**Source**: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

**Dataset Details**:
- **Institution**: Portuguese retail bank
- **Time Period**: May 2008 to June 2013
- **Total Records**: 41,188 phone contacts (bank-additional-full.csv)


## Project Structure

Link to GitHub With Jupiter notebook - 

## Analysis Workflow

### 1. Data Understanding & Exploration
- Dataset overview and structure
- Missing value analysis
- Class distribution analysis
- Feature correlation analysis
- Exploratory data analysis (EDA) 

### 2. Data Preprocessing
- Handling missing values
- Feature encoding (One-Hot Encoding for categorical variables)
- Feature scaling (StandardScaler for numerical features)
- Train-test split (80-20)

### 3. Baseline Model
- Dummy Classifier (stratified strategy)
- Establishes performance benchmark

### 4. Model Training & Comparison

Four classifiers were trained and evaluated:

1. **K-Nearest Neighbors (KNN)**
2. **Logistic Regression**
3. **Decision Tree Classifier**
4. **Support Vector Machine (SVM)**

### 5. Hyperparameter Tuning

GridSearchCV was used to optimize model performance:

- **KNN**: n_neighbors, weights, metric
- **Logistic Regression**: C, penalty, solver, class_weight
- **Decision Tree**: max_depth, min_samples_split, min_samples_leaf, class_weight
- **SVM**: C, kernel, gamma, class_weight

### 6. Model Evaluation

Models were evaluated using multiple metrics:
Accuracy,Precision,Recall,F1-Score and Training Time.

## Key Results

### Best Performing Model: Decision Tree Classifier

Performance Metrics (After HyperParameterTuning):
Test Accuracy: 90.2%
Precision: 65.3%
Recall: 45.8%
F1-Score: 0.5432

Improvement Over Baseline:
- F1-Score increased from 0.0000 (baseline) to 0.5432
- Significantly better balance between precision and recall
- Successfully identifies positive cases while maintaining accuracy

### Model Comparison Summary

| Model | Test Accuracy | Precision | Recall | F1-Score | Train Time |
|-------|--------------|-----------|--------|----------|------------|
| Decision Tree (Tuned) | 90.2% | 65.3% | 45.8% | 0.5432 | ~2s |
| Logistic Regression (Tuned) | 89.8% | 62.7% | 42.1% | 0.5051 | ~5s |
| KNN (Tuned) | 88.9% | 58.4% | 38.9% | 0.4662 | ~1s |
| SVM (Tuned) | 89.5% | 61.2% | 40.5% | 0.4881 | ~45s |
| Baseline (Dummy) | 88.7% | 0.0% | 0.0% | 0.0000 | <1s |

Best Model: Decision Tree (Tuned)

## Key Insights

### Technical Insights
1. Hyperparameter tuning significantly improved all models' performance
2. F1-score optimization is more appropriate than accuracy for imbalanced datasets
3. Class weight balancing helps models better handle class imbalance
4. Decision Trees performed best overall for this classification problem
5. All models successfully beat the naive baseline

### Business Value
Targeted Marketing: Can identify high-probability customers for term deposits
Cost Reduction: Reduce wasted marketing calls by focusing on likely subscribers
Improved Conversion: Optimize conversion rate beyond the ~11% baseline
Resource Optimization: Better allocation of marketing team resources
ROI Enhancement: Maximize return on investment for telemarketing campaigns

### Feature Importance (from EDA)
The most influential factors for term deposit subscription include:
- Economic indicators (Euribor rate, employment variation rate)
- Call direction (inbound vs outbound)
- Previous campaign outcome
- Contact duration
- Social and economic context variables