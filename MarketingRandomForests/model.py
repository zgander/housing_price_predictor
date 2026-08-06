import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load cleaned dataset
data = pd.read_csv("./dat/cleaned_marketing.csv")

# Drop index column if present
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])

df = pd.DataFrame(data)

x = df.drop("y", axis=1)
y = df["y"]

# Stratified Train-Test Split (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter Grid for Random Forest
param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2]
}

# 5-Fold Cross-Validation optimizing F1 Score for fast & robust evaluation
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(x_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation F1 Score:", grid_search.best_score_)

best_rf = grid_search.best_estimator_

# Predict on train and test sets
predictions_train = best_rf.predict(x_train)
predictions_test = best_rf.predict(x_test)

# Confusion Matrices
tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, predictions_train).ravel().tolist()
tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, predictions_test).ravel().tolist()

# Metrics Calculation Function
def calculate_metrics(tn, fp, fn, tp, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return acc, prec, rec, f1

train_acc, train_prec, train_rec, train_f1 = calculate_metrics(tn_train, fp_train, fn_train, tp_train, y_train, predictions_train)
test_acc, test_prec, test_rec, test_f1 = calculate_metrics(tn_test, fp_test, fn_test, tp_test, y_test, predictions_test)

print("\n--- Training Set Evaluation ---")
print(f"Training Confusion Matrix (TN, FP, FN, TP): {tn_train}, {fp_train}, {fn_train}, {tp_train}")
print(f"Training Accuracy Score: {train_acc:.6f}")
print(f"Training Precision: {train_prec:.6f}")
print(f"Training Recall: {train_rec:.6f}")
print(f"Training F1 Score: {train_f1:.6f}")

print("\n--- Test Set Evaluation ---")
print(f"Test Confusion Matrix (TN, FP, FN, TP): {tn_test}, {fp_test}, {fn_test}, {tp_test}")
print(f"Test Accuracy Score: {test_acc:.6f}")
print(f"Test Precision: {test_prec:.6f}")
print(f"Test Recall: {test_rec:.6f}")
print(f"Test F1 Score: {test_f1:.6f}")
