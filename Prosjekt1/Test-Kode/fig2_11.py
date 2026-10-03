# Bias-Variance-Tradeoff OLS

# First Fig 2.11 Test & Training MSE vs Poly.Deg 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def design_matrix(x, degree, intercept=True):
    # polynomial features [1, x, x^2, ..., x^degree] (drop the 1 if intercept=False)
    start = 0 if intercept else 1
    return np.vstack([x**p for p in range(start, degree + 1)]).T

rng = np.random.default_rng(2026)
n = 350
sigma = 0.5             # noise level
x = np.sort(rng.uniform(-1, 1, n))
y = runge(x) + rng.normal(0, sigma, n)

xx = np.linspace(-1, 1, 400)

maxdeg = 20
deg_V = np.arange(maxdeg + 1)
MSE_test = []
MSE_train = []
R2_V = []
Parameter_V = []
Prediction_V = []

for degree in range(maxdeg + 1):
    X = design_matrix(x, degree)
    X_train, X_test, y_train, y_test = train_test_split(X,
    y, test_size=0.3, random_state=2026)

    theta = np.linalg.pinv(X_train) @ y_train
    Parameter_V.append(theta)

    y_predict_test = X_test @ theta
    y_predict_train = X_train @ theta
    # Here i create the data for figure 2.11
    from sklearn.metrics import mean_squared_error
    mse_test = mean_squared_error(y_test, y_predict_test)
    MSE_test.append(mse_test)
    mse_train = mean_squared_error(y_train, y_predict_train)
    MSE_train.append(mse_train)


plt.plot(deg_V, MSE_test, label = "Test MSE")
plt.plot(deg_V, MSE_train, label = "Train MSE")
plt.title("Fig 2.11 Hastie et. al")
plt.xlabel("Model complexity")
plt.ylabel("Prediction error")
plt.legend()
plt.show()



