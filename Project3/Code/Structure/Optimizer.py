import numpy as np
# LLM declaration: NumPy provides the array operations used by the optimizer updates.

# [Changed 2026-11-17] All step_* methods now update the weights and moment arrays
# in place (W -= ..., m *= ...). Before, new arrays and lists were built in every
# minibatch. The step methods still return the layers, so the calling code is unchanged.
# NB: the layers passed in are modified directly.

class Optimizer:
    """Oppdaterer nettverkslag med plain gradient descent, Adam eller RMSprop."""

    def __init__(
        self, method, gamma,
        beta1=0.9, beta2=0.999, rho=0.99, eps=1e-8
    ):
        self.method = method.lower()
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.eps = eps

        self.state = None
        self.t = 0

    def zero_state(self, layers):
        """Lag nullstilte momentverdier for Adam og RMSprop."""
        return [
            (
                np.zeros_like(W),  # mW
                np.zeros_like(b),  # mb
                np.zeros_like(W),  # vW
                np.zeros_like(b),  # vb
            )
            for W, b in layers
        ]

    def initialize(self, layers):
        """Start en ny treningskjøring."""
        self.t = 0

        if self.method == "plain":
            self.state = None
        else:
            self.state = self.zero_state(layers)

    def step(self, layers, layer_grads, gamma=None):
        """Utfør én minibatchoppdatering."""
        if gamma is None:
            gamma = self.gamma

        self.t += 1

        if self.method == "plain":
            return self.step_plain(layers, layer_grads, gamma)

        if self.state is None:
            raise RuntimeError("Kjør optimizer.initialize(layers) før trening.")

        if self.method == "adam":
            return self.step_adam(layers, layer_grads, gamma)

        if self.method == "rmsprop":
            return self.step_rmsprop(layers, layer_grads, gamma)

        raise ValueError("method må være 'plain', 'adam' eller 'rmsprop'")

    def step_plain(self, layers, layer_grads, gamma):
        # [Changed 2026-11-17] In place, was a new list of (W - gamma*dW, b - gamma*db)
        for (W, b), (dW, db) in zip(layers, layer_grads):
            W -= gamma * dW
            b -= gamma * db
        return layers

    def step_rmsprop(self, layers, layer_grads, gamma):
        # [Changed 2026-11-17] In place. The formulas are unchanged:
        #   v = rho*v + (1-rho)*g^2,   P = P - gamma*g/(sqrt(v)+eps)
        # The same loop handles both W and b. np.asarray(dW) was removed, since the
        # gradients are now already NumPy arrays (see Act_Cost._xp).
        rho, eps = self.rho, self.eps
        for (W, b), (dW, db), (_, _, vW, vb) in zip(layers, layer_grads, self.state):
            for P, g, v in ((W, dW, vW), (b, db, vb)):
                v *= rho
                v += (1 - rho) * g * g
                P -= gamma * g / (np.sqrt(v) + eps)
        return layers

    def step_adam(self, layers, layer_grads, gamma):
        # [Changed 2026-11-17] In place. The formulas are unchanged:
        #   m = b1*m + (1-b1)*g,   v = b2*v + (1-b2)*g^2
        #   m_hat = m/c1, v_hat = v/c2,   P = P - gamma*m_hat/(sqrt(v_hat)+eps)
        # The same loop handles both W and b.
        b1, b2, eps = self.beta1, self.beta2, self.eps
        c1 = 1 - b1**self.t   # bias correction for m
        c2 = 1 - b2**self.t   # bias correction for v
        for (W, b), (dW, db), (mW, mb, vW, vb) in zip(layers, layer_grads, self.state):
            for P, g, m, v in ((W, dW, mW, vW), (b, db, mb, vb)):
                m *= b1
                m += (1 - b1) * g
                v *= b2
                v += (1 - b2) * g * g
                P -= gamma * (m / c1) / (np.sqrt(v / c2) + eps)
        return layers
