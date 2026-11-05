import numpy as np
# LLM declaration: NumPy creates one-hot targets and processes predicted probabilities.
from sklearn.base import BaseEstimator, ClassifierMixin
# LLM declaration: These scikit-learn base classes make the classifier compatible with estimators and model selection.
from sklearn.metrics import accuracy_score
# LLM declaration: Scikit-learn computes classification accuracy.
from .Act_Cost import ACTIVATIONS, cross_entropy, softmax
# LLM declaration: The classifier uses these activation functions and its cross-entropy objective.
from .NeuralNetwork import NeuralNetwork
# LLM declaration: The classifier constructs and uses the project neural network.
from .SGD import train_sgd
# LLM declaration: The classifier trains its network with the project's SGD implementation.

class NeuralNetClassifier(BaseEstimator, ClassifierMixin):
    "AI Declaration"
    def __init__(self, 
    hidden=(64,), activation="relu", 
    gamma=0.1, lmbd=0.0, 
    penalty=None, method = "plain",
    epochs=10, batch_size=5, # batch_size var 100 
    seed=2026):
        self.hidden = hidden
        self.activation = activation
        self.gamma = gamma
        self.lmbd = lmbd
        self.penalty = penalty 
        self.method = method
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y):
        """LLM Assistance: Creating the activation funcs in correct format"""
        hidden_func, hidden_der = ACTIVATIONS[self.activation]
        activation_funcs = [hidden_func] * len(self.hidden) + [softmax]
        activation_ders = [hidden_der] * len(self.hidden) + [None]

        self.classes_, class_indices = np.unique(y, return_inverse=True) # AI assisted
        n_classes = len(self.classes_) #AI assisted

        # Én rad per datapunkt, én kolonne per klasse.
        # AI assisted: Onehot encoding, Right form for cross entropy 
        targets = np.eye(n_classes)[class_indices] 
        
        self.network = NeuralNetwork(
            network_input_size=X.shape[1],
            layer_output_sizes = (*self.hidden, n_classes), # AI Assistance: Correction of sizes for one-hot
            activation_funcs = activation_funcs, 
            activation_ders = activation_ders,
            cost_fun = cross_entropy,
            cost_der = None,
            seed=self.seed
        )
        
        self.history, self.layer = train_sgd(self.network,
        X, targets, method = self.method, epochs = self.epochs,
        batch_size = self.batch_size, gamma = self.gamma, lmb = self.lmbd, 
        penalty = self.penalty, seed = self.seed)


        return self

    def predict_proba(self, X):
        # AI Declaration: GPT-61 Sol help
        return np.asarray(self.network.predict(X))

    def predict(self, X):
        # AI Declaration: GPT-61 Sol help
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[class_indices]

    def accuracy(self, predictions, targets):
        return accuracy_score(predictions, targets) # ski-kit function