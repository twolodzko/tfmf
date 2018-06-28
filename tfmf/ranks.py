
import numpy as np


def rank(data, axis=1):
    """Transform matrix to matrix of ranks along the axis

    Parameters
    ----------

    data : array
        Array to sort.

    axis : int, default : 1
        Rank the values along the axis.

    Returns
    -------

    array, shape (n_rows, n_cols)
        Ranked data array.
    """
    sidx = np.argsort(-data, axis=axis)
    m,n = data.shape
    out = np.empty((m,n), dtype=int)
    if axis == 1:
        out[np.arange(m)[:, None], sidx] = np.arange(n)
    else:
        out[sidx, np.arange(n)] = np.arange(m)[:, None]
    return out


def top_k_ranks(data, k=5, axis=1):
    """Return indexes of top k ranked columns or rows

    k : int, default : 5
        Limit results to top k ranks.

    axis : int, default : 1
        Rank the values along the axis.

    Returns
    -------

    array, shape (n_rows or n_cols, k)
        Ranked data array.
    """
    # see: https://stackoverflow.com/a/51081190/3986320

    k = min(k, data.shape[axis])
    ranked = rank(data, axis=axis)

    if axis == 1:
        out = np.zeros(shape=(data.shape[0], k), dtype=np.int32)
    else:
        out = np.zeros(shape=(k, data.shape[1]), dtype=np.int32)

    for r in range(k):
        idx = np.argwhere(ranked == r)
        if axis == 1:
            out[idx[:, 0], r] = idx[:, 1]
        else:
            out[r, idx[:, 1]] = idx[:, 0]

    return out


if __name__ == "__main__":

    # unit tests

    X = np.array([
        [1,2,30],
        [4,5,6],
        [90,8,7],
        [12,15,10]
    ])

    X_true_rank_ax1 = np.array([
        [2, 1, 0],
        [2, 1, 0],
        [0, 1, 2],
        [1, 0, 2]
    ])

    assert np.all(rank(X, axis=1) == X_true_rank_ax1)

    X_true_rank_ax0 = np.array([
        [3, 3, 0],
        [2, 2, 3],
        [0, 1, 2],
        [1, 0, 1]
    ])

    assert np.all(rank(X, axis=0) == X_true_rank_ax0)

    X_top2_ax1 = np.array([
        [2, 1],
        [2, 1],
        [0, 1],
        [1, 0],
    ])

    assert np.all(top_k_ranks(X, k=2, axis=1) == X_top2_ax1)

    X_top2_ax0 = np.array([
        [2, 3, 0],
        [3, 2, 3]
    ])

    assert np.all(top_k_ranks(X, k=2, axis=0) == X_top2_ax0)