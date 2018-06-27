import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


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

    mode : 'dok', 'csr', default : 'dok'
        Type of sparse matrix to be used. See scipy.sparse documentation for details.
    """

    if shape is None:
        n = np.max(rows) + 1
        k = np.max(cols) + 1
    
    if mode == 'dok':
        return coo_matrix((values, (rows, cols)), shape=(n, k)).todok()
    else:
        return csr_matrix((values, (rows, cols)), shape=(n, k))