class NeuralNetRegressor(BaseEstimator, RegressorMixin):
    """Your Project 2 network with the scikit-learn estimator interface."""

    def __init__(self, hidden=(50,), activation="sigmoid", 
    gamma=0.01, lmbd=0.0, penalty=None, method = "plain",
    epochs=set_epoch, batch_size=10, seed=2026):
        # store the hyperparameters under the same names; no computation here
        self.hidden = hidden
        self.activation = activation
        self.gamma = gamma
        self.lmbd = lmbd
        self.penalty = penalty
        self.method = method
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y, m_descent = "sgd"):
        # X has shape (n, 1), y shape (n,): build the layers from self.seed,
        # train with your SGD for self.epochs epochs, store the layers in self.layers_

        """LLM Assistance: Creating the activation funcs in correct format"""
        hidden_func, hidden_der = ACTIVATIONS[self.activation]
        activation_funcs = [hidden_func] * len(self.hidden) + [identity]
        activation_ders = [hidden_der] * len(self.hidden) + [identity_der]

        self.network_ = NeuralNetwork(
            network_input_size=X.shape[1],
            layer_output_sizes=(*self.hidden,1), # LLM Assisted - Shape error
            activation_funcs= activation_funcs,
            activation_ders = activation_ders,
            cost_fun=mse,
            cost_der=mse_der,
            seed=self.seed
        )

        if m_descent == "sgd":
            self.history, self.layer = train_sgd(self.network_, X, y, method = self.method,
                                epochs = self.epochs, batch_size = self.batch_size, 
                                gamma = self.gamma, lmb = self.lmbd, penalty=self.penalty, seed = self.seed)
        elif m_descent == "plain":
            self.history, self.layer = train_plain(self.network_, X, y, gamma = self.gamma, lmb = self.lmbd, penalty=self.penalty, epochs = self.epochs, seed = self.seed)
        else:
            raise ValueError("m_descent must be 'sgd' or 'plain'")
        return self

    def predict(self, X):
        """LM Assisted: Use of class functionality
           LM Assisted: Reshaping of output array to 1D for Scikit""" 
        # the network output as a 1D array of length len(X)
        predictions = self.network_.predict(X)
        predictions_1D = np.asarray(predictions).reshape(-1)
        return predictions_1D