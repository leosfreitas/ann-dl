"""The perceptron itself -- written from scratch, shared by both exercises.

Nothing here comes from a library beyond NumPy: the step activation, the
error-driven update and the stopping rule are all explicit.  Exercise 2 reuses
this module unchanged and only asks ``fit`` to keep a pocket copy, so the two
exercises genuinely run the same training code.

Conventions fixed by the statement:

* labels are ``{0, 1}`` (not ``{-1, +1}``), so the error term ``y - y_hat``
  takes values in ``{-1, 0, +1}`` and is zero exactly when the sample is right;
* the activation is a step with a ``>= 0`` threshold;
* the bias starts at ``0`` while the weights start at small non-zero values
  drawn by the caller, which is what makes the learning rate observable at all
  (see the note in ``fit``);
* one epoch is one sequential pass over the samples in the order given -- the
  updates are online, applied sample by sample, never batched.
"""

import numpy as np


def predict(X, w, b):
    """Step activation: 1 where w.x + b >= 0, else 0.  Vectorised over rows."""
    return (X @ w + b >= 0.0).astype(int)


def accuracy(X, y, w, b):
    return float(np.mean(predict(X, w, b) == y))


def fit(X, y, w0, eta=0.01, max_epochs=100, b0=0.0, pocket=False):
    """Train the perceptron and return everything the write-up has to report.

    The loop is deliberately the textbook one.  For each sample in turn we
    predict, and *only on a mistake* do we move the weights:

        w <- w + eta * (y - y_hat) * x
        b <- b + eta * (y - y_hat)

    A correct sample leaves the state untouched, which is the whole reason a
    separable dataset stops on its own: once no sample is misclassified, a full
    epoch produces zero updates and there is nothing left to change.  That is
    the first stopping rule; the second is the hard cap of ``max_epochs``, which
    is what actually fires when the data is not separable.

    ``pocket=True`` additionally remembers the best weights ever *seen*, scored
    on the full training set after **every update** -- Gallant's original pocket
    rule, not an end-of-epoch snapshot.  The distinction matters here: the
    samples arrive as one class stacked on the other, so every epoch ends just
    after a run of 1000 class-1 points has dragged the boundary clear off the
    data.  An end-of-epoch snapshot only ever sees that degenerate state and
    would report a pocket barely above 50%, missing the genuinely good weight
    vectors the run passes through in the middle of each pass.

    The running weights and the pocket weights are otherwise completely
    independent -- the pocket never feeds back into training, it only records.

    Returns a dict with the final state, the pocket state, and the per-epoch
    accuracy curves that Figures 3 and 6 are drawn from.
    """
    w = np.asarray(w0, dtype=float).copy()
    b = float(b0)

    curve = []        # accuracy of the running weights, after each epoch
    best_curve = []   # best-so-far accuracy, i.e. what the pocket holds
    updates_per_epoch = []

    best_w, best_b, best_acc, best_epoch = w.copy(), b, -1.0, 0
    best_update = 0
    converged = False
    epochs_run = 0
    total_updates = 0
    # (epoch, cumulative update index, accuracy) after every weight change --
    # recorded only under pocket, where that accuracy is computed anyway.
    trace = []

    if pocket:                           # the starting weights are a candidate too
        best_acc = accuracy(X, y, w, b)
        best_w, best_b, best_epoch = w.copy(), b, 0

    for epoch in range(1, max_epochs + 1):
        updates = 0
        for i in range(X.shape[0]):
            y_hat = 1 if (w @ X[i] + b) >= 0.0 else 0
            err = y[i] - y_hat
            if err != 0:                 # correct samples contribute nothing
                w += eta * err * X[i]
                b += eta * err
                updates += 1
                total_updates += 1
                if pocket:
                    # Strictly-greater, so the earliest weights attaining the
                    # best accuracy are the ones kept.
                    a = accuracy(X, y, w, b)
                    trace.append((epoch, total_updates, a))
                    if a > best_acc:
                        best_acc, best_w, best_b = a, w.copy(), b
                        best_epoch = epoch
                        best_update = total_updates

        epochs_run = epoch
        acc = accuracy(X, y, w, b)
        curve.append(acc)
        updates_per_epoch.append(updates)

        if not pocket and acc > best_acc:
            best_acc, best_w, best_b, best_epoch = acc, w.copy(), b, epoch
        best_curve.append(best_acc)

        if updates == 0:                 # a clean pass -- nothing left to learn
            converged = True
            break

    return {
        "w": w.copy(), "b": b,
        "acc": accuracy(X, y, w, b),
        "epochs": epochs_run,
        "converged": converged,
        "curve": curve,
        "best_curve": best_curve,
        "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
        "pocket_w": best_w, "pocket_b": best_b,
        "pocket_acc": best_acc, "pocket_epoch": best_epoch,
        "pocket_update": best_update,
        "trace": trace,
    }
