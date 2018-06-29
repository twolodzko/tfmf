
import numpy as np
from scipy.sparse import coo_matrix


def sparse_matrix(rows, cols, values, shape=None, mode='dok'):
    """Transform data into sparse matrix

    Create 'ijv' matrix X[rows, cols] = values.

    rows : array, shape (n_cells,)
        Row indexes.

    cols : array, shape (n_cells,)
        Column indexes.

    values : array, shape (n_cells,)
        Data.

    shape : tuple (n_rows, n_cols)
        Shape of the resulting matrix.

    mode : 'dok', 'csr', 'csc', 'coo', default : 'dok'
        Type of sparse matrix to be used. See scipy.sparse documentation for details.
    """
    if mode not in ['dok', 'csr', 'csc', 'coo']:
        raise ValueError

    if shape is None:
        n = np.max(rows) + 1
        k = np.max(cols) + 1
    else:
        n, k = shape
    
    mtx = coo_matrix((values, (rows, cols)), shape=(n, k))

    if mode == 'csr':    
        return mtx.tocsr()
    elif mode == 'csc':
        return mtx.tocsc()
    elif mode == 'dok':
        return mtx.todok()
    else:
        return mtx