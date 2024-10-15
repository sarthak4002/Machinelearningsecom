# SECOM Failure Prediction using Random Forest and PCA

## Project Overview

This project aims to predict system failures in the SECOM manufacturing dataset using machine learning techniques. The approach involves data preprocessing, class imbalance handling using SMOTE, dimensionality reduction with PCA, and classification using a Random Forest classifier. The model is tuned using GridSearchCV for optimal performance, and key metrics such as accuracy, confusion matrix, and ROC-AUC are evaluated.

## Dataset

- **Data**: SECOM manufacturing dataset (`secom.data`), containing sensor measurements from the production line.
- **Labels**: Binary labels (`secom_labels.data`) indicating whether a system failure occurred (`1` for failure, `-1` for non-failure).

Both data files are read using `pandas` and processed to handle missing values and class imbalance.

## Steps Involved

1. **Data Loading and Preprocessing**
   - Missing values in the dataset are handled using mean imputation.
   - The target labels are converted to binary format, where `-1` is replaced by `0` for non-failure.
   
2. **Handling Class Imbalance**
   - Class imbalance is addressed using the **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic samples for the minority class.

3. **Dimensionality Reduction**
   - **PCA (Principal Component Analysis)** is applied to reduce the dimensionality of the data, retaining components that explain the most variance (set to 50 components in this case).

4. **Model Selection and Hyperparameter Tuning**
   - A **Random Forest Classifier** is used to classify the data.
   - Hyperparameter tuning is performed using **GridSearchCV** to find the best parameters such as the number of estimators, max depth, and minimum samples split.

5. **Model Evaluation**
   - The performance of the model is evaluated using **accuracy**, **confusion matrix**, **classification report**, and the **ROC-AUC score**.
   - Feature importance is also plotted to understand which features contribute most to the prediction.

## Key Results

- **Accuracy**: The Random Forest model achieved an accuracy of `X%` on the test set.
- **ROC-AUC**: The model's ROC-AUC score was `X.XX`, indicating a good balance between true positive rate and false positive rate.
  
The project includes feature importance analysis to identify which sensors contributed the most to the predictions.

## Dependencies

To run the project, you will need the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `imblearn`
- `matplotlib`

## Evaluation Metrics

- Accuracy: Measures the overall correctness of the model.
- Confusion Matrix: Provides insight into true positives, false positives, true negatives, and false negatives.
- ROC-AUC: A performance measurement for classification problems at various threshold settings.

## Feature Importance

A bar plot is provided to visualize the importance of the top 20 features in the model. These features represent sensor readings that contribute the most to the prediction of system failures.

## ROC Curve

The Receiver Operating Characteristic (ROC) curve is plotted to illustrate the trade-off between the true positive rate and the false positive rate at various classification thresholds.

## Future Work

- Explore other machine learning models such as Gradient Boosting or XGBoost to improve performance.
- Further tune hyperparameters to maximize predictive accuracy.
- Experiment with different feature engineering techniques for better model interpretability.

## Random Forest Results
![Screenshot 2024-10-16 021258](https://github.com/user-attachments/assets/add9cec0-2763-4099-884d-f5c2c80e3d47)
![Screenshot 2024-10-16 021154](https://github.com/user-attachments/assets/e5c6e7d8-824c-4ff1-88e7-d813903728f5)

You can install the dependencies using:

```bash
pip install -r requirements.txt
