import numpy as np
# LLM declaration: NumPy handles the input conversion, feature construction, and scaling.
from sklearn.model_selection import train_test_split
# LLM declaration: Scikit-learn splits the data into training and test sets.

def Data_process(x, y,
    test_size = 0.3, Process=None):
    X = np.asarray(x).reshape(-1, 1) # make X a column vector for the network
    y = np.asarray(y).reshape(-1, 1) # make y a column vector for the network


    if Process == "Split and scale":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=2026
        )

        # Scale polynomial features, leaving the intercept column unchanged.
        mean = X_train[:, 1:].mean(axis=0)
        std = X_train[:, 1:].std(axis=0)
        std[std == 0] = 1

        X_train[:, 1:] = (X_train[:, 1:] - mean) / std
        X_test[:, 1:] = (X_test[:, 1:] - mean) / std

        # I choose not to scale the target.
        #y_mean = y_train.mean(axis=0)
        #y_train_c = y_train - y_mean

        return X_train, X_test, y_train, y_test #y_train_c, y_test, y_mean
    
    if Process == "Scale":
        mean = X[:, 1:].mean(axis=0) 
        std = X[:, 1:].std(axis=0)
        std[std == 0] = 1
        
        X[:, 1:] = (X[:, 1:] - mean) / std

        # I choose not to scale the target.
        #y_mean = y.mean(axis=0)
        #y_c = y - y_mean
        
        return X, y  #y_c, y_mean

    if Process == None:
        return X, y