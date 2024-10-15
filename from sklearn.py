from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load and preprocess the data
load_data = lambda: (pd.read_csv('secom.data', delim_whitespace=True, header=None),
                     pd.read_csv('secom_labels.data', delim_whitespace=True, header=None))

data, labels = load_data()

# Handle missing values by imputing the mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Replace -1 with 0 for binary classification in labels
labels[0] = labels[0].replace(-1, 0)
y = labels[0]

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(data_imputed, y)

# Split into training and test sets
split_data = lambda X, y: train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)  # Adjust the number of components based on variance explained
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', None]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_pca, y_train)

# Best model
best_rf_model = grid_search.best_estimator_

# Make predictions
y_pred = best_rf_model.predict(X_test_pca)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# ROC Curve and AUC
y_pred_proba = best_rf_model.predict_proba(X_test_pca)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Feature importance
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the top 20 features
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(20), importances[indices[:20]], align="center")
plt.xticks(range(20), [f'Feature_{i}' for i in indices[:20]], rotation=90)
plt.show()