import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def design_matrix(x, degree, intercept=True):
    # polynomial features [1, x, x^2, ..., x^degree] (drop the 1 if intercept=False)
    start = 0 if intercept else 1
    return np.vstack([x**p for p in range(start, degree + 1)]).T

rng = np.random.default_rng(2026)
n = 350
sigma = 0.5             # noise level fig 2.11
x = np.sort(rng.uniform(-1, 1, n))
y = runge(x) + rng.normal(0, sigma, n)


def bootstrap_bias_variance(n_bs, maxdeg):

    error = np.zeros(maxdeg)
    bias = np.zeros(maxdeg)
    variance = np.zeros(maxdeg)

    for degree in range(maxdeg):
        X = design_matrix(x, degree)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=2026
        )

        y_pred = np.zeros((y_test.shape[0], n_bs))
        for i in range(n_bs):
            X_bs, y_bs = resample(X_train, y_train)
            theta = np.linalg.pinv(X_bs) @ y_bs

            y_pred[:, i] = X_test @ theta

        # Compare each test target with every bootstrap prediction for that point.
        error[degree] = np.mean((y_test[:, None] - y_pred)**2)
        bias[degree] = np.mean((y_test - np.mean(y_pred, axis=1))**2)
        variance[degree] = np.mean(np.var(y_pred, axis=1))
    
    return error, bias, variance 
# maxdeg = 20 as in 2.11
# 40 bins
error40, bias40, variance40 = bootstrap_bias_variance(n_bs=40, maxdeg=20)
# 100 bins
error100, bias100, variance100 = bootstrap_bias_variance(n_bs=100, maxdeg=20)
# 400 bins
error300, bias300, variance300 = bootstrap_bias_variance(n_bs=300,maxdeg=20)

degree_vec = range(20)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

results = [
    (error40, bias40, variance40, 40),
    (error100, bias100, variance100, 100),
    (error300, bias300, variance300, 300)
]

for i, (error, bias, variance, n) in enumerate(results):

    ax[i].plot(degree_vec, error, label="Test error", color="blue")
    ax[i].scatter(degree_vec, error, color="blue")

    ax[i].plot(degree_vec, bias, label="Bias^2", color="orange")
    ax[i].scatter(degree_vec, bias, color="orange")

    ax[i].plot(degree_vec, variance, label="Variance", color="green")
    ax[i].scatter(degree_vec, variance, color="green")

    ax[i].axhline(y=sigma**2, color="black", linestyle="-", linewidth=1)

    ax[i].set_title(f"n_bins = {n}")
    ax[i].set_xlabel("Polynomial degree")
    ax[i].set_ylabel("Error")
    ax[i].legend()
    ax[i].set_ylim(0, 0.5)

plt.tight_layout()
plt.show()

# python doc/Projects/2026/Project1/bootstrap_bv.py