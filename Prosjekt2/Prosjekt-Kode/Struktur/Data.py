# The data of project 1, as a starting point for part b)
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2026)

def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

rng = np.random.default_rng(2026)      # fix the seed for reproducible results
n = 350 # 350 in Project 1
x = np.sort(rng.uniform(-1.0, 1.0, n))
sigma = 0.25 # 0.25 in Project 1
y = runge(x) + rng.normal(0.0, sigma, n)

# For the network the input is the raw x (one feature), samples in rows
X = x.reshape(-1, 1)
print(X.shape, y.shape)
plt.plot(x, runge(x), label="Runge function")
plt.scatter(x, y, s=10, label=f"noisy data, sigma={sigma}")
plt.legend(); plt.show()



from sklearn.model_selection import train_test_split

def Data_process(x, y, degree, 
    test_size = 0.3, Process="Split and scale"):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1,1)
    X = np.column_stack([x**power for power in range(degree + 1)])

    if Process == "Split and scale":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=2026
        )

        # Scale polynomial features, leaving the intercept column unchanged.
        mean = X_train[:, 1:].mean(axis=0)
        std = X_train[:, 1:].std(axis=0)
        mean[0] = 0
        std[std == 0] = 1

        X_train[:, 1:] = (X_train[:, 1:] - mean) / std
        X_test[:, 1:] = (X_test[:, 1:] - mean) / std

        y_mean = y_train.mean(axis=0)
        y_train_c = y_train - y_mean

        return X_train, X_test, y_train_c, y_test, y_mean
    
    if Process == "Scale":
        mean = X[:, 1:].mean(axis=0)
        std = X[:, 1:].std(axis=0)
        mean[0]=0
        std[std == 0] = 1
        
        X[:, 1:] = (X[:, 1:] - mean) / std

        y_mean = y.mean(axis=0)
        y_c = y - y_mean
        
        return X, y, y_mean