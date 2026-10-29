"""Exercise 1 -- one backpropagation step, checked numerically.

The derivation itself is done by hand in the report.  This script only redoes
the same arithmetic in NumPy so every hand-computed number can be checked, and
writes the values to ``results/mlp1.json``.  It is a checker, not a model: no
training loop, nothing from mlp.py.
"""

import json
import os
import numpy as np

RES = "../results"
os.makedirs(RES, exist_ok=True)

x = np.array([0.5, -0.2])
y = 1.0
W1 = np.array([[0.3, -0.1],
               [0.2, 0.4]])          # row i = weights into hidden neuron i
b1 = np.array([0.1, -0.2])
W2 = np.array([0.5, -0.3])
b2 = 0.2
eta = 0.3

# ------------------------------------------------------------------ A: forward
z1 = W1 @ x + b1
h1 = np.tanh(z1)
u2 = W2 @ h1 + b2
y_hat = np.tanh(u2)
L = (y - y_hat) ** 2

# ----------------------------------------------------------------- B: backward
dL_dyhat = -2.0 * (y - y_hat)
dL_du2 = dL_dyhat * (1.0 - y_hat ** 2)
dL_dW2 = dL_du2 * h1
dL_db2 = dL_du2
dL_dh1 = dL_du2 * W2
dL_dz1 = dL_dh1 * (1.0 - h1 ** 2)
dL_dW1 = np.outer(dL_dz1, x)
dL_db1 = dL_dz1

# ------------------------------------------------------------------- C: update
W2n = W2 - eta * dL_dW2
b2n = b2 - eta * dL_db2
W1n = W1 - eta * dL_dW1
b1n = b1 - eta * dL_db1

# Second forward pass with the updated parameters.
y_hat2 = np.tanh(W2n @ np.tanh(W1n @ x + b1n) + b2n)
L2 = (y - y_hat2) ** 2

out = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in {
    "z1": z1, "h1": h1, "u2": u2, "y_hat": y_hat, "L": L,
    "dL_dyhat": dL_dyhat, "dL_du2": dL_du2, "dL_dW2": dL_dW2, "dL_db2": dL_db2,
    "dL_dh1": dL_dh1, "dL_dz1": dL_dz1, "dL_dW1": dL_dW1, "dL_db1": dL_db1,
    "W2_new": W2n, "b2_new": b2n, "W1_new": W1n, "b1_new": b1n,
    "y_hat_after": y_hat2, "L_after": L2,
}.items()}

with open(os.path.join(RES, "mlp1.json"), "w") as f:
    json.dump(out, f, indent=2)

for k, v in out.items():
    print(f"{k:12s}", np.round(v, 6))
