import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

"""
# LLM declaration (Claude Code, 2026-11-19): helper functions for the two-stage search.
# search_table collects cv_results_ in a DataFrame with short column names,
# CV accuracy, its standard error, training accuracy and mean fit time.
# n_params counts the trainable weights and biases of an FFNN with the given hidden layers.
"""


def search_table(search):
    res = search.cv_results_
    df = pd.DataFrame([{key.split("__")[-1]: val for key, val in p.items()}
                       for p in res["params"]])
    df["val"] = res["mean_test_score"]
    df["val_se"] = res["std_test_score"] / np.sqrt(search.n_splits_ - 1)
    df["train"] = res["mean_train_score"]
    df["fit_time"] = res["mean_fit_time"]
    return df


def n_params(hidden, n_in=784, n_out=10):
    sizes = [n_in, *hidden, n_out]
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
