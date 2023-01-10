import numpy as np


def l2_normalize(x: np.ndarray,
                 axis: int = None,
                 out=None
                 ) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / norm
