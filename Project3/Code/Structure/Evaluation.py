"""
# LLM declaration (Claude Code, 2026-11-25): helper functions for the model comparison table.

Description
-----------
Evaluates fitted classifiers on the same test set with the same metrics, so the
methods (XGBoost, Random Forest, logistic regression, FFNN, ...) can be compared
side by side even though they are trained in different notebooks.

evaluate      computes test accuracy with its standard error, macro F1, macro one-vs-rest AUC and log loss
              for one fitted model and stores the result as one row in a shared
              CSV file (Project3/Results/model_comparison.csv). Rerunning with the
              same model name overwrites the old row.
load_table    reads the CSV back as a DataFrame, sorted by accuracy.
to_latex      returns the table as a LaTeX tabular (needs \\usepackage{booktabs}).

The model must have predict_proba and classes_ (sklearn-style), which holds for
RandomForestClassifier, XGBClassifier, LogisticRegression, MLPClassifier and
pipelines ending in one of these.

Usage
-----
    from Structure.Evaluation import evaluate, load_table, to_latex

    # In each notebook, after the final fit (always on the full 10 000-image test set):
    evaluate("XGBoost", final_model, x_test, y_test, n_train=len(x_train))
    evaluate("FFNN", search_class.best_estimator_, x_test_full, y_test_full, n_train=len(X_sub))

    # Anywhere, once all models are evaluated:
    df = load_table()
    print(df.to_string(index=False, float_format="%.4f"))
    print(to_latex(df))
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

# Structure -> Code -> Project3, so the file ends up in Project3/Results
# regardless of which notebook (Code/ or Test-code/) imports this module.
RESULTS = Path(__file__).resolve().parents[2] / "Results" / "model_comparison.csv"


def evaluate(name, model, X_test, y_test, n_train, path=RESULTS):
    """Compute test metrics for one fitted model and store them as a row in `path`."""
    proba = model.predict_proba(X_test)
    pred = model.classes_[proba.argmax(axis=1)]
    acc = accuracy_score(y_test, pred)
    row = {
        "Model": name,
        "Training size": n_train,
        "Accuracy": acc,
        # LLM declaration (Claude Code, 2026-11-25): binomial standard error of the test accuracy,
        # SE = sqrt(p(1-p)/n), used to judge whether differences between models exceed the uncertainty.
        "Accuracy SE": np.sqrt(acc * (1 - acc) / len(y_test)),
        "Macro F1": f1_score(y_test, pred, average="macro"),
        "Macro AUC": roc_auc_score(y_test, proba, multi_class="ovr", average="macro"),
        "Log loss": log_loss(y_test, proba),
    }

    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(df):
        df = df[df["Model"] != name]          # rerunning overwrites the model's old row
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return row


def load_table(path=RESULTS):
    """Read the comparison table, best accuracy first."""
    return pd.read_csv(path).sort_values("Accuracy", ascending=False).reset_index(drop=True)


def to_latex(df, caption="Test-set performance of all classifiers on Fashion-MNIST.",
             label="tab:model_comparison"):
    """Return the table as LaTeX code for the report."""
    return df.to_latex(index=False, float_format="%.4f", caption=caption, label=label)
