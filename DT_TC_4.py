#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import shap
from sklearn import tree

# Load
file_path = "E:/Research/PRODIGY_DS_03/Dataset/covid_19_clean_complete.csv"
clean = pd.read_csv(file_path)

# Info
print(clean.info())
print(clean.head())

# Numeric
numeric_columns = clean.select_dtypes(include=[np.number]).columns
imputer = KNNImputer(n_neighbors=5)
clean[numeric_columns] = imputer.fit_transform(clean[numeric_columns])

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

# Model
model = DecisionTreeRegressor(random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=5)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Initial Evaluation
print("Initial Model Evaluation:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Tree
plt.figure(figsize=(12, 8))
tree.plot_tree(model, filled=True, feature_names=features, rounded=True, max_depth=3)
plt.title('Simplified Decision Tree for Predicting Active Cases')
plt.show()

# Grid Search
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best Params
print("Best Parameters from GridSearchCV:", grid_search.best_params_)

# Best Model
best_model = grid_search.best_estimator_

# Best Predict
y_pred_best = best_model.predict(X_test)

# Tuned Evaluation
print("Tuned Model Evaluation:")
print("MSE:", mean_squared_error(y_test, y_pred_best))
print("MAE:", mean_absolute_error(y_test, y_pred_best))
print("R-squared:", r2_score(y_test, y_pred_best))

# Tuned Tree
plt.figure(figsize=(12,8))
tree.plot_tree(best_model, filled=True, feature_names=features, rounded=True, max_depth=3)
plt.title('Tuned Decision Tree for Predicting Active Cases')
plt.show()

# Performance
tuned_mae = mean_absolute_error(y_test, y_pred_best)
tuned_mse = mean_squared_error(y_test, y_pred_best)
tuned_r2 = r2_score(y_test, y_pred_best)

print("Tuned Model Performance Metrics:")
print(f"MAE: {tuned_mae}")
print(f"MSE: {tuned_mse}")
print(f"R-squared: {tuned_r2}")

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(best_model, X, y, cv=5, n_jobs=-1,
                                                         train_sizes=np.linspace(0.1, 1.0, 10),
                                                         scoring='neg_mean_squared_error')

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, -train_scores.mean(axis=1), label="Training Error", color='blue')
plt.plot(train_sizes, -test_scores.mean(axis=1), label="Test Error", color='red')
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Random Forest
rf_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest MSE:", mean_squared_error(y_test, rf_pred))
print("Random Forest R-squared:", r2_score(y_test, rf_pred))

# Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
print("Gradient Boosting MSE:", mean_squared_error(y_test, gb_pred))
print("Gradient Boosting R-squared:", r2_score(y_test, gb_pred))

# SHAP
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)
