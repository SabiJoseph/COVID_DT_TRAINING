#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load
file_path = "E:\Research\PRODIGY_DS_03\Dataset\cleaned_full_grouped.csv"
data = pd.read_csv(file_path)

# Features
X = data.drop(columns=['Date', 'High_Risk'])
y = data['High_Risk']

# Identify
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['bool']).columns

# Convert
X = X.astype({col: 'int' for col in categorical_cols})

# Transformers
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = SimpleImputer(strategy='most_frequent')

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Apply
X_imputed = preprocessor.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train
classifier = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, random_state=42)
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize
plt.figure(figsize=(12, 8))
plot_tree(classifier, feature_names=X.columns, class_names=['Low Risk', 'High Risk'], filled=True, proportion=True)
plt.title("Decision Tree Visualization")
plt.show()

# Confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
