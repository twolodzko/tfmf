
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm, trange
from scipy.sparse import coo_matrix, csr_matrix
import json

from .tf_model import TFModel


class MatrixFactorizer(BaseEstimator):
    
    """Matrix Factorizer
    
    Factorize the matrix R (n, k) into P (n, n_components) and Q (n_components, k) weights
    matrices:
    
        R[i,j] = P[i,:] * Q[:,j]
    
    Additional intercepts mu, bi, bj can be included, leading to the following model:
    
        R[i,j] = mu + bi[i] + bj[j] + P[i,:] * Q[:,j]
        
    The model is commonly used for collaborative filtering in recommender systems, where
    the matrix R contains of ratings by n users of k products. When users rate products
    using some kind of rating system (e.g. "likes", 1 to 5 stars), we are talking about
    explicit ratings (Koren et al, 2009). When ratings are not available and instead we
    use indirect measures of preferences (e.g. clicks, purchases), we are talking about
    implicit ratings (Hu et al, 2008). For implicit ratings we use modified model, where
    we model the indicator variable:
    
        D[i,j] = 1 if R[i,j] > 0 else 0
        
    and define additional weights:
    
        C[i,j] = 1 + alpha * R[i, j]
        
    or log weights:
    
        C[i,j] = 1 + alpha * log(1 + R[i, j])
        
    The model is defined in terms of minimizing the loss function (squared, logistic) between
    D[i,j] indicators and the values predicted using matrix factorization, where the loss is
    weighted using the C[i,j] weights (see Hu et al, 2008 for details). When using logistic
    loss, the predictions are passed through the sigmoid function to squeze them into the
    (0, 1) range.
    
    Parameters
    ----------
    
    n_components : int, default : 5
        Number of latent components to be estimated. The estimated latent matrices P and Q
        have (n, n_components) and (n_components, m) shapes subsequently.
    
    n_iter : int, default : 500
        Number of training epochs, the actual number of iterations is n_samples * n_epoch.
        
    batch_size : int, default : 500
        Size of the random batch to be used during training. The batch_size is the number of
        cells that are randomly sampled from the factorized matrix.
    
    learning_rate : float, default : 0.01
        Learning rate parameter.
    
    regularization_rate : float, default : 0.02
        Regularization parameter.
        
    alpha : float, default : 1.0
        Weighting parameter in matrix factorization with implicit ratings.
    
    implicit : bool, default : False
        Use matrix factorization with explicit (default) or implicit ratings. 
    
    loss : 'squared', 'logistic', default: 'squared'
        Loss function to be used. For implicit=True 'logistic' loss may be preferable.
        
    log_weights : bool, default : None
        Used only when implicit=True, then it defaults to log_weights=True, so log weighting
        is used in the loss function instead of standard weights (log_weights=False).
        
    fit_intercepts : bool, default : True
        When set to True, the mu, bi, bj intercepts are fitted, otherwise
        only the P and Q latent matrices are fitted.
        
    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as initialization,
        otherwise, just erase the previous solution.
        
    optimizer : 'Adam', 'Ftrl', default : 'Adam'
        Optimizer to be used, see TensorFlow documentation for more details.
        
    random_state : int, or None, default : None
        The seed of the pseudo random number generator to use when shuffling the data. If int,
        random_state is the seed used by the random number generator.
        
    show_progress : bool, default : False
        Show the progress bar.
    
    Examples
    --------
    
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tfmf import MatrixFactorizer, sparse_matrix
    >>> user_id = [0,0,1,1,2,2]
    >>> movie_id = [0,1,2,0,1,2]
    >>> rating = [1,1,2,2,3,3]
    >>> X = sparse_matrix(user_id, movie_id, rating)
    >>> mf = MatrixFactorizer(n_components=2, n_iter=100, batch_size=6, random_state=42, show_progress=False)
    >>> mf.partial_fit(X)
    MatrixFactorizer(alpha=1.0, batch_size=6, fit_intercepts=True, implicit=False,
            learning_rate=0.01, log_weights=None, loss='squared',
            n_components=2, n_iter=100, optimizer='Adam', random_state=42,
            regularization_rate=0.02, show_progress=False, warm_start=False)
    >>> X_full = np.array([[i,j] for i in range(3) for j in range(3)])
    >>> np.reshape(mf.predict(X_full[:,0], X_full[:,1]), (3,3))
    array([[1.1241099 , 0.4444648 , 0.5635694 ],
           [1.6370661 , 1.1460071 , 1.2965162 ],
           [0.55132747, 2.502296  , 2.5540314 ]], dtype=float32)
    
    References
    ----------
    
    Koren, Y., Bell, R., & Volinsky, C. (2009).
    Matrix factorization techniques for recommender systems. Computer, 42(8).
    
    Yu, H. F., Hsieh, C. J., Si, S., & Dhillon, I. (2012, December).
    Scalable coordinate descent approaches to parallel matrix factorization for recommender systems.
    In Data Mining (ICDM), 2012 IEEE 12th International Conference on (pp. 765-774). IEEE.
    
    Hu, Y., Koren, Y., & Volinsky, C. (2008, December).
    Collaborative filtering for implicit feedback datasets.
    In Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on (pp. 263-272). IEEE.
    
    """ 
        
    def __init__(self, n_components=5, n_iter=500, batch_size=500, learning_rate=0.01,
                 regularization_rate=0.02, alpha=1.0, implicit=False, loss='squared',
                 log_weights=None, fit_intercepts=True, warm_start=False, optimizer='Adam',
                 random_state=None, show_progress=True):
        
        self.n_components = n_components
        self.shape = (None, None, self.n_components)
        self._data = None
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.learning_rate = float(learning_rate)
        self.alpha = float(alpha)
        self.regularization_rate = float(regularization_rate)
        self.implicit = implicit
        self.loss = loss

        if implicit and log_weights is None:
            self.log_weights = True
        else:
            self.log_weights = log_weights
        
        self.fit_intercepts = fit_intercepts
        self.optimizer = optimizer
        self.random_state = random_state
        self.warm_start = warm_start
        self.show_progress = show_progress
        
        np.random.seed(self.random_state)
        self._fresh_session()
    
        
    def _fresh_session(self):
        # reset the session, to start from the scratch        
        self._tf = None
        self.history = []
    
    
    def _tf_init(self, shape=None):
        # define the TensorFlow model and initialize variables, session, saver
        if shape is None:
            shape = self.shape
        self._tf = TFModel(shape=self.shape, learning_rate=self.learning_rate,
                           alpha=self.alpha, regularization_rate=self.regularization_rate,
                           implicit=self.implicit, loss=self.loss, log_weights=self.log_weights,
                           fit_intercepts=self.fit_intercepts, optimizer=self.optimizer,
                           random_state=self.random_state)
    
            
    def _get_batch(self, data, batch_size=1):
        # create single batch for training
        
        batch_rows = np.random.randint(self.shape[0], size=batch_size)
        batch_cols = np.random.randint(self.shape[1], size=batch_size)

        # extract elements from scipy.sparse matrix
        batch_vals = data[batch_rows, batch_cols].A.flatten()
        
        return batch_rows, batch_cols, batch_vals
    
      
    def init_with_shape(self, n, k):
        '''Manually initialize model for given shape of factorized matrix

        n, k : int
            Shape of the factorized matrix.
        '''
        self.shape = (int(n), int(k), int(self.n_components))
        self._tf_init(self.shape)
    
    
    def fit(self, sparse_matrix):
        '''Fit the model
        
        Parameters
        ----------
        
        sparse_matrix : sparse-matrix, shape (n_users, n_items)
            Sparse matrix in scipy.sparse format, can be created using sparse_matrix
            function from this package.
        '''
        if not self.warm_start:
            self._fresh_session()
        return self.partial_fit(sparse_matrix)
    
    
    def partial_fit(self, sparse_matrix):
        '''Fit the model
        
        Parameters
        ----------
        
        sparse_matrix : sparse-matrix, shape (n_users, n_items)
            Sparse matrix in scipy.sparse format, can be created using sparse_matrix
            function from this package.
        '''
                    
        if self._tf is None:
            self.init_with_shape(*sparse_matrix.shape)
        
        for _ in trange(self.n_iter, disable=not self.show_progress):
            batch_rows, batch_cols, batch_vals = self._get_batch(sparse_matrix, self.batch_size)
            loss_value = self._tf.train(batch_rows, batch_cols, batch_vals)
            self.history.append(loss_value)
                
        return self
    
    
    def predict(self, rows, cols):
        '''Predict using the model
        
        Parameters
        ----------
        
        rows : array, shape (n_samples,)
            Row indexes.

        cols : array, shape (n_samples,)
            Column indexes.
        '''
        
        return self._tf.predict(rows, cols)
