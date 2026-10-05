from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from volgate.risk.dist import empirical_var_es, student_t_var_es


@dataclass
class FitResult:
    params: dict
    z: np.ndarray | None
    nu: float | None
    converged: bool


class VolModel(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> FitResult: ...

    def filter(self, fit: FitResult, data: pd.DataFrame) -> np.ndarray: ...


def tail_var_es(fit: FitResult, sigma: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if fit.nu is not None:
        return student_t_var_es(sigma, fit.nu, alpha)
    return empirical_var_es(fit.z, sigma, alpha)
