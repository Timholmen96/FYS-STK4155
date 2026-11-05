import numpy as np
# LLM declaration: NumPy provides array operations used by the network and its gradients.
import jax
# LLM declaration: JAX provides automatic differentiation for jax_gradient.
import jax.numpy as jnp
# LLM declaration: JAX NumPy provides array operations used in forward propagation.
from sklearn.metrics import accuracy_score
# LLM declaration: Scikit-learn computes classification accuracy.
from .Act_Cost import softmax
# LLM declaration: The network uses softmax to detect the classification output layer.

class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
        seed,
    ):
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders 
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.rng = np.random.default_rng(seed) # Reproducability

        # create layers = [(W , b)] with random normal weights and biases
        layers = []

        i_size = network_input_size
        for layer_output_size in layer_output_sizes:
            W = self.rng.standard_normal((i_size, layer_output_size)) / np.sqrt(i_size) # dele på sqrt(i_size)? Ja for batched
            b = np.zeros(layer_output_size) + 0.01 #np.random.randn(layer_output_size)
            layers.append((W, b))

            i_size = layer_output_size
        
        self.layers = layers

    def predict(self, inputs):
        # Simple feed forward pass
        a = jnp.asarray(inputs) 
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a

    def cost(self, inputs, targets):
        predictions = self.predict(inputs)
        #print("Prediksjoner:", predictions.shape)
        #print("Fasit:", targets.shape)
        #print("Kostfunksjon:", self.cost_fun.__name__)

        mse = self.cost_fun(predictions,targets)
        return mse

    def _feed_forward_saver(self, inputs):
        layer_inputs = []
        zs = []
        a = jnp.asarray(inputs)
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            #print("a",a.shape)
            #print("W",W.shape)
            #print("b", b.shape)
            z = a @ W + b
            zs.append(z)
            a = activation_func(z)
        return layer_inputs, zs, a 

    def compute_gradient_standard(self, inputs, targets):
        # Backpropagation
        # dC_dW = dC_da @ da_dz @ dz_dW
        # dC_db = dC_da @ da_dz @ dz_db
        layer_inputs, sz, predict = self._feed_forward_saver(
            inputs)
        layer_grads = [() for layer in self.layers]

        for l in reversed(range(len(self.layers))):
            layer_input, z, activation_der = layer_inputs[l], sz[l], self.activation_ders[l]

            if l == (len(self.layers) - 1): # last layer
                dC_da = self.cost_der(predict, targets)
            else:
                (W, b) = self.layers[l + 1]
                #print("layer", l)
                #print("predictions:", predict.shape)
                #print("targets:", targets.shape)
                #print("dC_dz:", dC_dz.shape)
                #print("W:", W.shape)
                dC_da = dC_dz @ (W.T) 

            dC_dz = dC_da * activation_der(z) # Hadamard
            dC_dW = layer_input.T @ dC_dz
            dC_db = np.sum(dC_dz, axis=0) # what happened here?

            layer_grads[l] = (dC_dW, dC_db)
            #print("layer", l)
            #print("predictions:", predict.shape)
            #print("targets:", targets.shape)
            #print("dC_dz:", dC_dz.shape)
        return layer_grads

    def compute_gradient_softmax(self, inputs, targets):
        "Exercicesweek42.ipynb"
        layer_inputs, sz, predict = self._feed_forward_saver(inputs)
        layer_grads = [None] * len(self.layers) # AI assisted for simplicity GPT LUNA 6.1
        n = inputs.shape[0]

        for l in reversed(range(len(self.layers))):
            layer_input = layer_inputs[l]

            if l == len(self.layers) - 1:
                # Softmax + mean cross entropy
                dC_dz  = (predict - targets)/n
            else:
                (W, b) = self.layers[l + 1] # W_next, b_next
                z = sz[l]
                activation_der = self.activation_ders[l]
                dC_dz = (dC_dz @ W.T) * activation_der(z)
        
            dC_dW = layer_input.T @ dC_dz
            dC_db = np.sum(dC_dz, axis = 0)

            layer_grads[l] = (dC_dW, dC_db)

        return layer_grads

    def compute_gradient(self, inputs, targets):
        "AI Declaration: Created with help of GPT-6.1 Sol for SGD function compute_gradient"
        if self.activation_funcs[-1] is softmax:
            return self.compute_gradient_softmax(inputs, targets)

        return self.compute_gradient_standard(inputs, targets)

    
    def accuracy(self, predictions, targets):
        "Exerciseweek42: Classification"
        one_hot_predictions = np.zeros(predictions.shape)

        for i, prediction in enumerate(predictions):
            one_hot_predictions[i, np.argmax(prediction)] = 1
        return accuracy_score(one_hot_predictions, targets) # ski-kit function

    # These last two methods are not needed in the project, but they can be nice to have!
    # This should be External! But i can try use plain gradient
    # BEWARE: Using this function will change the local variable self.layers in the object
    def update_weights_plain(self, layer_grads, gamma=0.05):
        self.layers = [
        (W - gamma * W_g, b - gamma * b_g)
        for (W, b), (W_g, b_g) in zip(self.layers, layer_grads)]


    # jax solution
    # The first one has a layers parameter so that you can use jax.grad on it.
    
    def jax_compliant_predict(self, layers, inputs):
        a = jnp.asarray(inputs)

        for (W, b), activation_func in zip(layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)
        return a

    def jax_cost(self, layers, inputs, targets):
        predictions = self.jax_compliant_predict(layers, inputs)
        mse = self.cost_fun(predictions, targets)
        return mse

    def jax_gradient(self, inputs, targets):
        grads = jax.grad(self.jax_cost,argnums = 0)(self.layers, inputs, targets)
        return grads