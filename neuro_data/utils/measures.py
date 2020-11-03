import numpy as np

def corr(y1,y2, axis=-1, ddof=0, eps=1e-8, **kwargs):
    """
    Compute the correlation between two matrices along certain dimensions. 

    Args:
        y1:      first matrix
        y2:      second matrix
        axis:    dimension along which the correlation is computed.
        ddof:    delta degrees of freedom for standard deviation
        eps:     offset to the standard deviation to make sure the correlation is well defined (default 1e-8)
        **kwargs passed to final mean of standardized y1 * y2

    Returns: correlation vector

    """
    y1 = (y1 - y1.mean(axis=axis, keepdims=True))/(y1.std(axis=axis, keepdims=True, ddof=ddof) + eps)
    y2 = (y2 - y2.mean(axis=axis, keepdims=True))/(y2.std(axis=axis, keepdims=True, ddof=ddof) + eps)
    return (y1 * y2).mean(axis=axis, **kwargs)
