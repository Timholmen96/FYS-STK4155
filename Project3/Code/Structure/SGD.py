import numpy as np
# LLM declaration: NumPy converts training data, shuffles sample indices, and applies regularization.
from .Optimizer import Optimizer
# LLM declaration: The SGD training function updates network layers with Optimizer.

def step_length(t, t0, t1):
    """A decaying learning-rate schedule."""
    return t0 / (t + t1)


# [Changed 2026-11-17] Added: the L1/L2 code was moved out of the loop into this function.
# The gradients are now updated in place (dW += ...) instead of building new lists.
# Safe, because the gradient arrays are computed fresh in every batch.
"AI Declaration: Help from codex GPT-LUNA6.1 for task 2b) L1 and L2 placement"
def _add_penalty(layers, layer_grads, penalty, lmb):
    """Add the L1/L2 term to the weight gradients (biases are not penalized)."""
    if penalty is None or lmb == 0.0:
        if penalty not in (None, "l1", "l2"):
            raise ValueError("penalty må være 'l1', 'l2' eller None")
        return layer_grads
    #L2 norm - Ridge
    if penalty == "l2":
        for (W, _), (dW, _) in zip(layers, layer_grads):
            dW += (2 * lmb) * W
        return layer_grads
    #L1 norm - Lasso
    if penalty == "l1":
        for (W, _), (dW, _) in zip(layers, layer_grads):
            dW += lmb * np.sign(W)
        return layer_grads
    raise ValueError("penalty må være 'l1', 'l2' eller None")


# [Changed 2026-11-17] Added: only used for float32.
def _flush_subnormals(arrays, tiny):
    """Set |x| < tiny to 0. In float32 the weights/moments drift into subnormal
    numbers (~1e-40), and matrix products with them are ~30x slower on CPU."""
    for A in arrays:
        A[np.abs(A) < tiny] = 0.0


def train_sgd(
    network, X, y, method="adam",
    epochs=10, batch_size=5, gamma=0.001, lmb = 0.0, penalty ="l2", schedule=None, seed=2026,
    dtype=np.float64  # [Changed 2026-11-17] Added: new parameter, float32 trains faster
):
    "LLM Assisted: Using the class netowrk"
    rng = np.random.default_rng(seed)

    # [Changed 2026-11-17] was np.asarray(X) / np.asarray(y).
    # Converts once to the chosen dtype. Keep the default float64 when you
    # compare against jax.grad to ~1e-15.
    X = np.ascontiguousarray(X, dtype=dtype)
    y = np.ascontiguousarray(y, dtype=dtype)
    # [Changed 2026-11-17] Added: the weights get the same dtype (astype makes a copy).
    network.layers = [(W.astype(dtype), b.astype(dtype)) for W, b in network.layers]
    n = len(X)

    # Bruk av klasse
    optimizer = Optimizer(method=method, gamma=gamma)
    optimizer.initialize(network.layers)
    # [Changed 2026-11-17] Removed: the unused variables state and history_test.

    t = 0
    history = []

    for epoch in range(epochs):
        # Bland rekkefølgen på datapunktene for hver epoke.
        # [Changed 2026-11-17] The whole set is shuffled once per epoch, and the minibatches
        # are now cheap slices instead of an index copy (X[batch]) in every batch.
        # rng.permutation(n) is drawn exactly as before, so the batches are identical.
        indices = rng.permutation(n)
        X_shuf = X[indices]
        y_shuf = y[indices]

        # Minibatch
        for start in range(0, n, batch_size):
            X_batch = X_shuf[start:start + batch_size]
            y_batch = y_shuf[start:start + batch_size]

            # Hent gradientene fra nettverksobjektet.
            layer_grads = network.compute_gradient(X_batch, y_batch)
            # [Changed 2026-11-17] L1/L2 moved to _add_penalty (see above).
            layer_grads = _add_penalty(network.layers, layer_grads, penalty, lmb)

            # Tell minibatchoppdateringer. Adam bruker t til korreksjon
            # for oppstartsfasen.
            # [Changed 2026-11-17] Bug fix: the scheduled step length was computed but never
            # passed to optimizer.step, so the schedule had no effect. It is passed now.
            step_gamma = gamma if schedule is None else step_length(t, *schedule)
            t += 1
            network.layers = optimizer.step(network.layers, layer_grads, gamma=step_gamma)

        # [Changed 2026-11-17] Added: remove subnormal numbers once per epoch (float32 only).
        if dtype == np.float32:
            tiny = np.finfo(np.float32).tiny
            _flush_subnormals([P for layer in network.layers for P in layer], tiny)
            if optimizer.state is not None:
                _flush_subnormals([A for st in optimizer.state for A in st], tiny)

        # Kostnaden etter at hele treningssettet har vært gjennom én gang.
        if epoch % 50 == 0 or epoch == epochs - 1:
            history.append(float(network.cost(X, y)))

    return history, network.layers
