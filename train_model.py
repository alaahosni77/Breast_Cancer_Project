# %%
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
print("Libraries imported successfully")

# %%
df = pd.read_csv("processed_breast_cancer.csv")
print("First 5 rows:")
print(df.head())
print("\nDataset shape:", df.shape)


# %%
X = df.drop("target", axis=1)
y = df["target"]
print("Features shape:", X.shape)
print("Target shape:", y.shape)


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# %%
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("Folders created successfully")

# %%
mlflow.set_tracking_uri(f"file:///{os.getcwd()}/mlruns")
mlflow.set_experiment("Breast Cancer Classification")

print("MLflow initialized")


# %%
def evaluate_model(model, model_name):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n===== {model_name} =====")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = f"plots/{model_name}_cm.png"
    plt.savefig(cm_path)
    plt.close()

    #ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{model_name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    roc_path = f"plots/{model_name}_roc.png"
    plt.savefig(roc_path)
    plt.close()

    
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_test,
        y_prob
    )

    plt.figure()
    plt.plot(recall_curve, precision_curve)
    plt.title(f"{model_name} Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    pr_path = f"plots/{model_name}_pr.png"
    plt.savefig(pr_path)
    plt.close()

    return accuracy, precision, recall, f1, cm_path, roc_path, pr_path


print("Evaluation function ready")


# %%
with mlflow.start_run(run_name="Logistic_Regression"):

    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)

    acc, prec, rec, f1, cm, roc, pr = evaluate_model(
        lr_model,
        "LogisticRegression"
    )

    mlflow.log_param("model", "LogisticRegression")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.log_artifact(cm)
    mlflow.log_artifact(roc)
    mlflow.log_artifact(pr)

    mlflow.sklearn.log_model(
        lr_model,
        "logistic_regression_model"
    )

print("Logistic Regression completed")


# %%
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_

print("Best Parameters:")
print(grid_search.best_params_)


# %%
with mlflow.start_run(run_name="Random_Forest"):

    acc, prec, rec, f1, cm, roc, pr = evaluate_model(
        best_rf_model,
        "RandomForest"
    )

    mlflow.log_param("model", "RandomForest")
    mlflow.log_params(grid_search.best_params_)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.log_artifact(cm)
    mlflow.log_artifact(roc)
    mlflow.log_artifact(pr)

    mlflow.sklearn.log_model(
        best_rf_model,
        "random_forest_model"
    )

print("Random Forest completed")

# %%
joblib.dump(best_rf_model, "model.pkl")
joblib.dump(best_rf_model, "models/random_forest_model.joblib")

print("Model saved successfully")

# %%
print("\nProject completed successfully")
print("Ready for MLflow UI")
