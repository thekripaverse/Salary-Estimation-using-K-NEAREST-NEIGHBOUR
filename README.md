# Salary Estimation Using K-Nearest Neighbors

**Machine Learning–Based Salary Prediction System**

This project implements a **K-Nearest Neighbors (KNN) regression model** to estimate salaries based on multiple professional and demographic factors such as years of experience, education level, and job role. The system demonstrates how distance-based learning can be applied to real-world regression problems using Python and scikit-learn.

---

## Overview

Salary prediction is a common real-world regression problem in machine learning. This project uses the KNN algorithm to predict an individual’s expected salary by analyzing similarities between feature vectors in the dataset. The model is trained on structured tabular data, scaled for optimal performance, and evaluated using standard regression metrics.

---

## Key Features

* Salary prediction based on multiple professional attributes
* Implementation of KNN for regression
* Feature scaling and normalization
* Hyperparameter tuning to identify the optimal K value
* Performance evaluation using error metrics
* Visualization of prediction performance

---

## Technology Stack

* Python
* NumPy
* Pandas
* scikit-learn
* Matplotlib

---

## Methodology

1. Data preprocessing and feature selection
2. Feature scaling and normalization
3. Dataset splitting into training and testing sets
4. Model training using KNN regression
5. Hyperparameter tuning to determine the optimal number of neighbors
6. Model evaluation using regression metrics
7. Visualization of results

---

## K-Nearest Neighbors Algorithm

KNN is a supervised, non-parametric algorithm that predicts outputs based on the similarity between data points.

For regression tasks:

* Distances between the input and all training points are computed.
* The K closest neighbors are selected.
* The predicted salary is the average of the neighbors’ salary values.

Model performance is evaluated using metrics such as **Mean Squared Error (MSE)** and overall predictive accuracy.

---

## Results

* Optimal K value: 6
* Model accuracy: 80%
* The model demonstrates effective salary trend estimation with properly scaled features.

---

## Installation and Execution

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
python knn_salary_prediction.py
```

---

## Applications

* Salary trend analysis
* HR analytics and decision support
* Entry-level machine learning demonstrations
* Regression modeling practice

---

## Future Enhancements

* Integration with a web interface (Streamlit or Flask)
* Inclusion of more features and real-world datasets
* Comparison with other regression models
* Cross-validation and advanced evaluation metrics

---
