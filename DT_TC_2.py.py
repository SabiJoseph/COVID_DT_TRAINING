#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

# Load
file_path = "E:/Research/PRODIGY_DS_03/Dataset/covid_19_clean_complete.csv"
clean = pd.read_csv(file_path)

# Info
print(clean.info())
print(clean.head())

# Columns
print("Columns in the dataset:", clean.columns)

# Missing
numeric_columns = clean.select_dtypes(include=[np.number]).columns
clean[numeric_columns] = clean[numeric_columns].fillna(clean[numeric_columns].mean())

# Categorical
categorical_columns = clean.select_dtypes(exclude=[np.number]).columns
for col in categorical_columns:
    clean[col] = clean[col].fillna(clean[col].mode()[0])

# Encode
label_encoder = LabelEncoder()
clean['WHO Region'] = label_encoder.fit_transform(clean['WHO Region'])

# Features
features = ['Lat', 'Long', 'Confirmed', 'Deaths', 'Recovered', 'WHO Region']
target = 'Active'

# Split
X = clean[features]
y = clean[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train
model = DecisionTreeRegressor(random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=5)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Initial Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Tree
plt.figure(figsize=(12,8))
tree.plot_tree(model, filled=True, feature_names=features, rounded=True, max_depth=3)  # Simplified
plt.title('Simplified Decision Tree for Predicting Active Cases')
plt.show()

# Importance
feature_importance = model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=features, y=feature_importance)
plt.title('Feature Importance in Predicting Active Cases')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# True vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('True vs Predicted Active Cases')
plt.xlabel('True Active Cases')
plt.ylabel('Predicted Active Cases')
plt.show()

# Errors
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals, color='red')
plt.axhline(0, color='blue', linestyle='--')
plt.title('Prediction Errors (True vs Residuals)')
plt.xlabel('True Active Cases')
plt.ylabel('Prediction Error')
plt.show()

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1,
                                                         train_sizes=np.linspace(0.1, 1.0, 10),
                                                         scoring='neg_mean_squared_error')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, -train_scores.mean(axis=1), label="Training Error", color='blue')
plt.plot(train_sizes, -test_scores.mean(axis=1), label="Test Error", color='red')
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
