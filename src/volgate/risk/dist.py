import numpy as np
from scipy import stats


def alpha_tag(alpha: float) -> str:
    return f"a{round(alpha * 1000):03d}"


def student_t_var_es(sigma, nu: float, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Lower-tail VaR and ES for sigma * (unit-variance Student-t with nu dof)."""
    if nu <= 2:
        raise ValueError(f"nu must exceed 2, got {nu}")
    q = stats.t.ppf(alpha, nu)
    scale = np.sqrt((nu - 2.0) / nu)
    es_std = -(nu + q**2) / (nu - 1.0) * stats.t.pdf(q, nu) / alpha
    sigma = np.asarray(sigma, dtype=float)
    return sigma * q * scale, sigma * es_std * scale


def empirical_var_es(z, sigma, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Lower-tail VaR and ES for sigma * z, z drawn from the empirical residuals."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    needed = 2 * int(np.ceil(1.0 / alpha))
    if z.size < needed:
        raise ValueError(f"need at least {needed} residuals, got {z.size}")
    q = np.quantile(z, alpha)
    es = z[z <= q].mean()
    sigma = np.asarray(sigma, dtype=float)
    return sigma * q, sigma * es
