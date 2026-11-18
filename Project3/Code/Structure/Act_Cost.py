import numpy as np
# LLM declaration: NumPy is used for array operations in activation and cost functions.
import jax
# [Changed 2026-11-17] Added: jax is imported so that _xp can recognise JAX arrays.
import jax.numpy as jnp
# LLM declaration: JAX NumPy provides array operations for the differentiable activation and cost functions.

# [Changed 2026-11-17] Added _xp (speed-up).
# Every function now picks its array library from the input:
#   NumPy arrays (training)            -> NumPy, fast for small arrays.
#   JAX arrays/tracers (jax.grad check) -> jax.numpy, still differentiable.
# Before, all functions used jnp. Eager JAX calls have a fixed overhead each,
# and the training loop makes thousands of them, which slowed down training.
def _xp(z):
    return jnp if isinstance(z, jax.Array) else np


def sigmoid(z):
    xp = _xp(z)  # [Changed 2026-11-17] was jnp.exp
    return 1 / (1 + xp.exp(-z))
def sigmoid_der(z):
    s = sigmoid(z)
    return s*(1 - s)
# Alternatively sigmoid_der = jax.grad(jnp.sum(sigmoid(z)))
# Identity
def identity(z):
    return z

def identity_der(z):
    return _xp(z).ones_like(z)  # [Changed 2026-11-17] was np.ones_like

# Defining some activation functions
def ReLU(z):
    # [Changed 2026-11-17] was jnp.where(z > 0, z, 0.0). The int 0 keeps the dtype (float32 stays float32)
    return _xp(z).where(z > 0, z, 0)
# Derivative of the ReLU function
def ReLU_der(z):
    # [Changed 2026-11-17] was jnp.where(z > 0, 1.0, 0.0), which always gave float64
    # and made float32 training slow (mixed float32/float64 matrix products).
    return (z > 0).astype(z.dtype)

def mse(predict, target):
    xp = _xp(predict)  # [Changed 2026-11-17] was jnp.mean
    return xp.mean((predict - target) ** 2)
def mse_der(predict, target): # Derivative of cost function
    return 2*(predict - target)/predict.size # Mean brings a factor of 1/N, derivative brings factor 2
# Alternatively: mse_der = jax.grad(mse(predict, target))

def softmax(z):
    """Eq. (8.75) with the shift of Eq. (5.30): rows are samples, columns classes."""
    xp = _xp(z)  # [Changed 2026-11-17] was np; now also works with jax.grad
    e = xp.exp(z - xp.max(z, axis=1, keepdims=True)) # subtract max value for stability
    return e / xp.sum(e, axis=1, keepdims=True)

def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time."""
    xp = _xp(z)  # [Changed 2026-11-17] was jnp
    e_z = xp.exp(z - xp.max(z))
    return e_z / xp.sum(e_z)

def cross_entropy(predict, target):
    # Classification cost_fun
    # [Changed 2026-11-17] was jnp.log(predict). Clipping prevents log(0) = -inf -> NaN.
    xp = _xp(predict)
    return -xp.mean(
        xp.sum(target * xp.log(xp.clip(predict, 1e-12, 1.0)), axis=1)
    )


ACTIVATIONS = {"sigmoid": (sigmoid, sigmoid_der),
               "relu": (ReLU, ReLU_der),
               "identity": (identity, identity_der),
               "softmax": (softmax, None)}


OUTPUT = {"mse": identity, "binary": sigmoid, "softmax": softmax}

#set_epoch = 100
