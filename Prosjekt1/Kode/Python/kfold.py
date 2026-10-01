# Cross validation functionality of Scikit-learn
# Kfold + cross_val_score or cross_validate

# Eveluate against the MSE function from the test folds
# For the OLS analysis of parts a) and c) as a 
# function of the polynomial degree. 

# Try k = 5 and k = 10
# Compare MSE from cross-validation with the one you got from
# your bootstrap code in part c)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils import resample



def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def design_matrix(x, degree, intercept=True):
    # polynomial features [1, x, x^2, ..., x^degree] (drop the 1 if intercept=False)
    start = 0 if intercept else 1
    return np.vstack([x**p for p in range(start, degree + 1)]).T

rng = np.random.default_rng(2026)
n = 400
sigma = 0.8                                  # noise level: explore it!
x = np.sort(rng.uniform(-1, 1, n))
y = runge(x) + sigma*rng.normal(0, sigma, n)


def runge_data(n=100, degree=6, noise=0.1, seed=2026):
    """Runge function 1/(1+25x^2) on [-1,1], standardised polynomial features, centred y."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    y = 1.0 / (1.0 + 25.0 * x**2) + noise * rng.standard_normal(n)
    X = np.column_stack([x**k for k in range(1, degree + 1)])
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_norm, y - y.mean()


def Kfold_CV(k, deg, model, lmb=0): # OLS med mindre lmb blir gitt verdi
    X = design_matrix(x, deg)
    kfold = KFold(n_splits=k, shuffle=True, random_state=2026)
    score_KFold = []
    
    for train_ind, test_ind in kfold.split(X):
        x_train, x_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

        # Skalering
        # Beregn gjennomsnitt og standardavvik fra treningsdata
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)

        # Behold konstantkolonnen intercept: (1 - 0) / 1 = 1
        mean[0] = 0
        std[0] = 1

        X_train_norm = (x_train - mean) / std
        X_test_norm = (x_test - mean) / std
        # Trenger ikke sentrere y da design matrisen inneholder intercept

        I = np.eye(X_train_norm.shape[1])
        I[0,0] = 0.0 # Vil ikke straffe intercept i Ridge regresjon

        if model == "OLS" or lmb == 0:
            theta = np.linalg.pinv(X_train_norm) @ y_train
        elif model == "Ridge":
            theta = (
                np.linalg.pinv(X_train_norm.T @ X_train_norm + lmb * I)
                @ X_train_norm.T @ y_train)
        elif model == "Lasso":
            # Lasso regression is not implemented manually here, as it requires iterative optimization.
            raise NotImplementedError("Manual Lasso regression is not implemented.")

        y_predict = X_test_norm @ theta 
        mse = np.mean((y_test - y_predict)**2)
        score_KFold.append(mse)
    mse_Kfold = np.mean(score_KFold)
    mse_std = np.std(score_KFold, ddof=1)
    return mse_Kfold, mse_std


def sklearn_kfold(folds, degree, model, lmb=None):
    X = design_matrix(x, degree)
    cv = KFold(n_splits=folds, shuffle=True, random_state=2026)

    if model == "OLS" or lmb is None:
        # X already includes the constant column.
        model = LinearRegression(fit_intercept=False)
    elif model == "Ridge":
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=lmb, fit_intercept=True, solver="svd")
        )
    elif model == "Lasso":
        model = make_pipeline(StandardScaler(),Lasso(alpha=lmb/2, fit_intercept=True, max_iter=10000, tol=1e-6))

    fold_mse = -cross_val_score(
        model, X, y,
        cv=cv,
        scoring="neg_mean_squared_error",
        error_score="raise"
    )

    return fold_mse.mean(), fold_mse.std(ddof=1)



degrees = np.arange(21)
lambdas = [1e-2, 1e-1, 1.0, 1e1]

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

for row, folds in enumerate([5, 10]):
    # OLS
    manual = np.array([
        Kfold_CV(folds, degree, "OLS")[0]
        for degree in degrees
    ])
    sklearn_mse = np.array([
        sklearn_kfold(folds, degree, "OLS")[0]
        for degree in degrees
    ])

    ax = axes[row, 0]
    ax.plot(degrees, manual, label="Manual")
    ax.plot(degrees, sklearn_mse, "x--", label="Scikit-learn")
    ax.set_title(f"OLS: {folds}-fold CV")
    print(
        f"OLS, k={folds}: maximum absolute MSE difference = "
        f"{np.max(np.abs(manual - sklearn_mse)):.3e}"
    )

    # Ridge: one curve per lambda
    ax = axes[row, 1]
    for lmb in lambdas:
        manual = np.array([
            Kfold_CV(folds, degree, "Ridge", lmb)[0]
            for degree in degrees
        ])
        sklearn_mse = np.array([
            sklearn_kfold(folds, degree, "Ridge", lmb)[0]
            for degree in degrees
        ])

        line, = ax.plot(
            degrees, sklearn_mse,
            label=rf"Scikit-learn $\lambda={lmb:g}$"
        )
        ax.plot(
            degrees, manual, "x", color=line.get_color(),
            label="Manual" if lmb == lambdas[0] else "_nolegend_"
        )
        print(
            f"Ridge, k={folds}, lambda={lmb:g}: "
            f"maximum absolute MSE difference = "
            f"{np.max(np.abs(manual - sklearn_mse)):.3e}"
        )

    ax.set_title(f"Ridge: {folds}-fold CV")

for ax in axes.flat:
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Mean test-fold MSE")
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()