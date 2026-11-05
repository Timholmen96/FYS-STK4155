import numpy as np
# LLM declaration: NumPy converts training data and applies regularization operations.
from .Optimizer import Optimizer
# LLM declaration: The plain-gradient training function updates network layers with Optimizer.

def train_plain(
    network, X, y, gamma=0.001, lmb = 0.0, penalty = None, epochs=10, seed = 2026
    ):
    rng = np.random.default_rng(seed)
    inputs = np.asarray(X)
    targets = np.asarray(y)
    
    optimizer = Optimizer(method="plain", gamma=gamma)
    optimizer.initialize(network.layers)
    state = None
    history = []
    

    for i in range(epochs):
        layer_grads = network.compute_gradient(inputs, targets)
        
        "Cope from SGD_train()"
        if penalty == "l2":
            layer_grads = [
            (dW + 2 * lmb * W, db)
            for (W, b), (dW, db) in zip(network.layers, layer_grads)]
        elif penalty == "l1":
            layer_grads = [
            (dW + lmb * np.sign(W), db)
            for (W, b), (dW, db) in zip(network.layers, layer_grads)
            ]
        elif penalty is not None:
            raise ValueError("penalty må være 'l1', 'l2' eller None")
        
        network.layers = optimizer.step(network.layers, layer_grads)
        if epoch % 50 == 0 or epoch == epochs - 1:
            epoch_cost = float(network.cost(X, y))
            history.append(epoch_cost)
            

    return history, network.layers