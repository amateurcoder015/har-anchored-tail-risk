import numpy as np


def fz0_loss(y, var, es, alpha: float) -> np.ndarray:
    """FZ0 joint VaR/ES loss (Patton, Ziegel & Chen 2019), lower tail.

    L = -1/(alpha*es) * 1{y <= var} * (var - y) + var/es + log(-es) - 1
    """
    y = np.asarray(y, dtype=float)
    var = np.asarray(var, dtype=float)
    es = np.asarray(es, dtype=float)
    if np.any(es >= 0):
        raise ValueError("ES must be negative")
    hit = (y <= var).astype(float)
    return -hit * (var - y) / (alpha * es) + var / es + np.log(-es) - 1.0
