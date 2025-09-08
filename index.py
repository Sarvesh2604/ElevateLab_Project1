import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import shap
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# Dataset Loading
df = pd.read_csv("IBM HR.csv")
print("Dataset Shape:", df.shape)
print(df.head())

# EDA
# Attrition Count
sns.countplot(x="Attrition", data=df, palette="Set2")
plt.title("Employee Attrition Distribution")
plt.show()

# Department wise Attrition
plt.figure(figsize=(8,4))
sns.countplot(x="Department", hue="Attrition", data=df, palette="Set1")
plt.title("Attrition by Department")
plt.show()

# Monthly Income vs Attrition
plt.figure(figsize=(6,4))
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df, palette="coolwarm")
plt.title("Monthly Income vs Attrition")
plt.show()

# Data Preprocessing
df = df.drop(columns=["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"])

# Encoding Variables
cat_cols = df.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Features and Targets
X = df.drop("Attrition", axis=1)
y = df["Attrition"]   # already encoded (0 = No, 1 = Yes)

# Train Test Splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model Building
# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Decision Tree
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

print("\n=== Decision Tree ===")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# Tree Model
importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": tree_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=importances.head(15), palette="viridis")
plt.title("Top 15 Important Features (Decision Tree)")
plt.show()

# Logistic Regression Confusion Matrix
disp_log = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_log),
                                  display_labels=log_model.classes_)
disp_log.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Decision Tree Confusion Matrix
disp_tree = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_tree),display_labels=tree_model.classes_)
disp_tree.plot(cmap="Greens")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

results = {
    "Model": ["Logistic Regression", "Decision Tree"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_tree)
    ]
}
report_df = pd.DataFrame(results)

report_df.to_csv("Model_Accuracy_Report.csv", index=False)

log_report = classification_report(y_test, y_pred_log, output_dict=True)
tree_report = classification_report(y_test, y_pred_tree, output_dict=True)

pd.DataFrame(log_report).to_csv("Logistic_Regression_Report.csv")
pd.DataFrame(tree_report).to_csv("Decision_Tree_Report.csv")

# SHAP
explainer = shap.TreeExplainer(tree_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")

sample = X_test.iloc[0:1]

shap_values_single = explainer.shap_values(sample)

shap.force_plot(
    explainer.expected_value[1],
    shap_values_single[1],
    sample
)
