"""A multi-layer perceptron written from scratch -- shared by Exercises 2, 3, 4.

Nothing here comes from a library beyond NumPy: the activations, the losses,
the forward pass, the backward pass and the gradient-descent update are all
explicit.  Exercises 2, 3 and 4 import this one module and call the same
``forward`` / ``backward`` / ``step`` functions; only the arguments change.

Conventions:

* ``sizes`` is the full list of layer widths, input first and output last, e.g.
  ``[2, 8, 1]`` or ``[4, 16, 16, 3]`` -- nothing about depth or width is
  hard-coded;
* every hidden layer uses ``tanh``;
* the output layer is a sigmoid when it has one unit (binary cross-entropy)
  and a softmax when it has more (categorical cross-entropy);
* training is full-batch gradient descent, so one epoch is one update.

Why one backward pass serves both tasks: for sigmoid + binary cross-entropy
*and* for softmax + categorical cross-entropy, the gradient of the mean loss
with respect to the output pre-activation ``u`` collapses to the same thing,

    dL/du = (y_hat - y) / N,

with ``y`` a column of 0/1 in the binary case and a one-hot matrix in the
multi-class case (the derivation is in the report, Section 3B).  Everything
upstream of ``u`` is ordinary chain rule through tanh layers and does not care
which output it is feeding.
"""

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------- activations
def tanh(z):
    return np.tanh(z)


def tanh_prime_from_output(h):
    """d tanh(z)/dz written in terms of h = tanh(z): 1 - h^2."""
    return 1.0 - h ** 2


def sigmoid(z):
    # Clipping keeps exp() finite; it changes nothing at the precision we use.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def softmax(z):
    # Subtracting the row max is the standard overflow guard; softmax is
    # invariant to adding a constant to every logit of a row.
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------- init
def init_params(sizes, rng):
    """Xavier/Glorot-uniform weights, zero biases.

    ``W[l]`` has shape (fan_in, fan_out), so a layer is ``X @ W + b`` with one
    sample per row.  The Glorot range keeps tanh pre-activations near the
    linear part of the curve at the start, where gradients are largest.
    """
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        W = rng.uniform(-limit, limit, size=(fan_in, fan_out))
        b = np.zeros(fan_out)
        params.append([W, b])
    return params


def n_params(sizes):
    return int(sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:])))


# -------------------------------------------------------------------- forward
def forward(params, X):
    """Return the list of layer outputs ``[X, h1, ..., y_hat]``.

    The cache is just the activations: tanh's derivative can be written from
    its own output, and the output layer's gradient never needs ``u``.
    """
    acts = [X]
    h = X
    for W, b in params[:-1]:
        h = tanh(h @ W + b)
        acts.append(h)
    W, b = params[-1]
    u = h @ W + b
    y_hat = sigmoid(u) if W.shape[1] == 1 else softmax(u)
    acts.append(y_hat)
    return acts


def loss(y_hat, Y):
    """Mean cross-entropy.  Binary when ``y_hat`` has one column, else categorical.

    ``Y`` must have the same shape as ``y_hat``: a 0/1 column, or one-hot rows.
    """
    p = np.clip(y_hat, EPS, 1.0 - EPS)
    if y_hat.shape[1] == 1:
        return float(-np.mean(Y * np.log(p) + (1.0 - Y) * np.log(1.0 - p)))
    return float(-np.mean(np.sum(Y * np.log(p), axis=1)))


# ------------------------------------------------------------------- backward
def backward(params, acts, Y):
    """Gradients of the mean loss for every (W, b), output layer first.

    ``delta`` is dL/d(pre-activation) of the layer being processed.  It starts
    as (y_hat - Y)/N at the output and is pushed back one layer at a time:

        dW     = h_prev^T @ delta
        db     = sum of delta over the batch
        dh     = delta @ W^T
        delta' = dh * (1 - h_prev^2)          (tanh)
    """
    N = Y.shape[0]
    delta = (acts[-1] - Y) / N
    grads = [None] * len(params)
    for l in range(len(params) - 1, -1, -1):
        W, _ = params[l]
        h_prev = acts[l]
        grads[l] = [h_prev.T @ delta, delta.sum(axis=0)]
        if l > 0:
            delta = (delta @ W.T) * tanh_prime_from_output(h_prev)
    return grads


# --------------------------------------------------------------------- update
def step(params, grads, eta):
    """Plain gradient descent, in place: theta <- theta - eta * dL/dtheta."""
    for (W, b), (dW, db) in zip(params, grads):
        W -= eta * dW
        b -= eta * db


# ------------------------------------------------------------------ utilities
def one_hot(y, k):
    Y = np.zeros((y.size, k))
    Y[np.arange(y.size), y] = 1.0
    return Y


def targets(y, n_out):
    """Integer labels -> the target matrix the loss expects."""
    return y.reshape(-1, 1).astype(float) if n_out == 1 else one_hot(y, n_out)


def predict(params, X):
    y_hat = forward(params, X)[-1]
    if y_hat.shape[1] == 1:
        return (y_hat[:, 0] >= 0.5).astype(int)
    return y_hat.argmax(axis=1)


def accuracy(params, X, y):
    return float(np.mean(predict(params, X) == y))


def fit(sizes, X, y, eta, epochs, rng):
    """Initialise and train an MLP; return the parameters and the loss curve.

    One epoch = one forward pass over the whole training set, one backward
    pass, one update.  The loss recorded for an epoch is the one measured on
    the forward pass *before* that epoch's update.
    """
    params = init_params(sizes, rng)
    Y = targets(y, sizes[-1])
    history = np.empty(epochs)
    for epoch in range(epochs):
        acts = forward(params, X)
        history[epoch] = loss(acts[-1], Y)
        step(params, backward(params, acts, Y), eta)
    return params, history
