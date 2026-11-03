def train_plain(
    network, X, y, gamma=0.001, lmb = 0.0, penalty = None, epochs=set_epoch, seed = 2026
    ):
    rng = np.random.default_rng(seed)
    inputs = np.asarray(X)
    targets = np.asarray(y)
    
    optimizer = Optimizer(method="plain", gamma=gamma)
    optimizer.initialize(network.layers)
    state = None
    history = []
    

    for i in range(epochs):
        layers_grad = network.compute_gradient(inputs, targets)
        network.layers = optimizer.step(network.layers, layers_grad)
        
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
        
        "Bør jeg flytte denne ut som SGD?"
        if i % 50 == 0 or i == epochs - 1:
            predictions = network.predict(inputs)
            history.append(
                (i, float(network.cost(inputs, targets)), 
                network.accuracy(predictions,targets)))
            

    return network.layers, history