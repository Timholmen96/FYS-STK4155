import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils import resample
import pandas as pd

def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def design_matrix(x, degree, intercept=True):
    # polynomial features [1, x, x^2, ..., x^degree] (drop the 1 if intercept=False)
    start = 0 if intercept else 1
    return np.vstack([x**p for p in range(start, degree + 1)]).T

rng = np.random.default_rng(2026)
n = 400
sigma = 0.25                                 # noise level: explore it!
x = np.sort(rng.uniform(-1, 1, n))
y = runge(x) + rng.normal(0, sigma, n)




def sklearn_kfold(folds, degree, model, alpha=None):
    X = design_matrix(x, degree)
    cv = KFold(n_splits=folds, shuffle=True, random_state=2026)

    if model == "OLS" and alpha is None:
        # X already includes the constant column.
        model = LinearRegression(fit_intercept=False)
    elif model == "Ridge": # alpha = lam
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha, fit_intercept=True, solver="svd")
        )
    elif model == "Lasso": # alpha = lam / 2
        model = make_pipeline(
            StandardScaler(),
            Lasso(alpha=alpha, fit_intercept=True, max_iter=10000, tol=1e-6)
        )

    fold_mse = -cross_val_score(
        model, X, y,
        cv=cv,
        scoring="neg_mean_squared_error",
        error_score="raise"
    )

    return fold_mse.mean(), fold_mse.std(ddof=1)

maxdeg = 15
k = [5, 10]
alpha = np.logspace(-3, 0, 4)  # alpha values for Ridge and Lasso
deg = [2, 5, 8, 10, 15, 20] # polynomial degrees to test


results = []

for folds in k:
    for degree in deg:
        # OLS har ingen regulariseringsparameter.
        mse, std = sklearn_kfold(folds, degree, "OLS")

        results.append({
            "Modell": "OLS",
            "Folds": folds,
            "Grad": degree,
            "Alpha": np.nan,
            "MSE": mse,
            "Std": std,
        })

        # Ridge og Lasso undersøkes for hver alpha.
        for method in ["Ridge", "Lasso"]:
            for a in alpha:
                mse, std = sklearn_kfold(
                    folds, degree, method, a
                )

                results.append({
                    "Modell": method,
                    "Folds": folds,
                    "Grad": degree,
                    "Alpha": a,
                    "MSE": mse,
                    "Std": std,
                })

results_df = pd.DataFrame(results)
print(results_df.head(10))
print(results_df)


def plot_results(results_df):
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6), sharex=True, sharey=True
    )

    # Samme alpha får samme farge i begge figurene og metodene.
    alpha_values = sorted(results_df["Alpha"].dropna().unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alpha_values)))
    alpha_colors = dict(zip(alpha_values, colors))

    for ax, folds in zip(axes, [5, 10]):
        fold_results = results_df[results_df["Folds"] == folds]

        # OLS har ingen alpha og vises derfor bare én gang.
        ols = fold_results[
            fold_results["Modell"] == "OLS"
        ].sort_values("Grad")

        ax.plot(
            ols["Grad"], ols["MSE"],
            color="black", marker="o", linewidth=2,
            label="OLS",
        )

        # Farge viser alpha; linjestil viser regresjonsmetode.
        for method, style, marker in [
            ("Ridge", "-", "o"),
            ("Lasso", "--", "s"),
        ]:
            model_results = fold_results[
                fold_results["Modell"] == method
            ]

            for a, group in model_results.groupby("Alpha"):
                group = group.sort_values("Grad")

                ax.plot(
                    group["Grad"],
                    group["MSE"],
                    color=alpha_colors[a],
                    linestyle=style,
                    marker=marker,
                    label=f"{method}, alpha={a:g}",
                )
                ax.legend(fontsize=8, ncol=2)

        ax.set(
            title=f"{folds}-fold kryssvalidering",
            xlabel="Polynomgrad",
            ylabel="CV-MSE",
        )
        ax.set_xticks(sorted(fold_results["Grad"].unique()))
        ax.grid(alpha=0.3)

    # Felles forklaring for begge plottene.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )

    fig.tight_layout()
    plt.show()


plot_results(results_df)


# Finn raden med lavest MSE for hvert antall folds.
best_indices = results_df.groupby("Folds")["MSE"].idxmin()
best_models = results_df.loc[best_indices].sort_values("Folds")

for _, row in best_models.iterrows():
    alpha_text = (
        "ikke relevant" if row["Modell"] == "OLS"
        else f"{row['Alpha']:.6g}"
    )

    print(
        f"k = {int(row['Folds'])}: "
        f"{row['Modell']}, "
        f"grad = {int(row['Grad'])}, "
        f"alpha = {alpha_text}, "
        f"CV-MSE = {row['MSE']:.6f}, "
        f"fold-standardavvik = {row['Std']:.6f}"
    )