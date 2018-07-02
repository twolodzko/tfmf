
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from tqdm import trange
from scipy.sparse import coo_matrix, csr_matrix

from .tf_model import TFModel
from .sparse_matrix import sparse_matrix


class MatrixFactorizer(BaseEstimator):
   
   """Matrix Factorizer
   
   Factorize the matrix R (n, k) into P (n, n_components) and Q (n_components, k) weights
   matrices:
   
      R[i,j] = P[i,:] * Q[:,j]
   
   Additional intercepts b0, bi, bj can be included, leading to the following model:
   
      R[i,j] = b0 + bi[i] + bj[j] + P[i,:] * Q[:,j]
      
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
   >>> mf.predict_all().A
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

      if loss not in ['squared', 'logistic']:
         raise ValueError("use 'squared' or 'logistic' loss")

      if optimizer not in ['Adam', 'Ftrl']:
         raise ValueError("use 'Adam' or 'Ftrl' optimizer")
      
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
   
   
   def _tf_init(self):
      # define the TensorFlow model and initialize variables, session, saver
      self._tf = TFModel(shape=self.shape, learning_rate=self.learning_rate,
                         alpha=self.alpha, regularization_rate=self.regularization_rate,
                         implicit=self.implicit, loss=self.loss, log_weights=self.log_weights,
                         fit_intercepts=self.fit_intercepts, optimizer=self.optimizer,
                         random_state=self.random_state)
   
         
   def _batch_generator(self, data, size=1, nonzero=False):
      if nonzero:
         # for explicit ratings
         rows, cols = data.nonzero()
         while True:
            idx = np.random.randint(len(rows), size=size)
            yield rows[idx], cols[idx], data[rows[idx], cols[idx]].A.flatten()
      else:
         # for implicit ratings
         while True:
            rows = np.random.randint(self.shape[0], size=size)
            cols = np.random.randint(self.shape[1], size=size)
            vals = data[rows, cols].A.flatten()
            yield rows, cols, vals


   def save(self, path):
      """Save model

      Parameters
      ----------

      path : str
         Directory where model files are saved.
      """
      self._tf.save(path)


   def restore(self, path):
      """Restore model from saved files

      Parameters
      ----------

      path : str
         Directory where model files can be found.

      Examples
      --------
 
      >>> import tempfile
      >>> import numpy as np
      >>> import pandas as pd
      >>> from tfmf import MatrixFactorizer, sparse_matrix
      >>> user_id = [0,0,0,1,1,1,2,2,2]
      >>> movie_id = [0,1,2,0,1,2,0,1,2]
      >>> rating = [0,1,1,2,2,3,3,4,4]
      >>> X = sparse_matrix(user_id, movie_id, rating)
      >>> mf = MatrixFactorizer(n_components=2, n_iter=2500, batch_size=9, show_progress=True,
                          regularization_rate=0.1, fit_intercepts=False, random_state=42)
      >>> mf.fit(X)
      MatrixFactorizer(alpha=1.0, batch_size=9, fit_intercepts=False,
             implicit=False, learning_rate=0.01, log_weights=None,
             loss='squared', n_components=2, n_iter=2500, optimizer='Adam',
             random_state=42, regularization_rate=0.1, show_progress=True,
             warm_start=False)
      >>> mf.predict_all().A
      array([[0.36638957, 0.82455045, 0.7515532 ],
            [1.8798509 , 2.2950716 , 2.664069  ],
            [2.6847882 , 3.658589  , 4.039307  ]], dtype=float32)
      >>> tmpdir = tempfile.gettempdir()
      >>> mf.save(tmpdir + '/tfmf')
      >>> mf = MatrixFactorizer(n_components=2, n_iter=2500, batch_size=9, show_progress=True,
                          regularization_rate=0.1, fit_intercepts=False, random_state=42)
      >>> mf.init_with_shape(3, 3)
      >>> mf.predict_all().A
      array([[ 1.11211761e-04,  1.85020617e-04,  5.69926306e-05],
            [-7.57966773e-05, -1.24432714e-04, -4.61529817e-05],
            [ 1.14005406e-05,  1.75371242e-05,  1.21055045e-05]], dtype=float32)
      >>> mf.restore(tmpdir + '/tfmf')
      >>> mf.predict_all().A
      array([[0.36638957, 0.82455045, 0.7515532 ],
            [1.8798509 , 2.2950716 , 2.664069  ],
            [2.6847882 , 3.658589  , 4.039307  ]], dtype=float32)
      """
      self._tf.restore(path)
   
     
   def init_with_shape(self, n, k):
      """Manually initialize model for given shape of factorized matrix

      n, k : int
         Shape of the factorized matrix.
      """
      self.shape = (int(n), int(k), int(self.n_components))
      self._tf_init()
   
   
   def fit(self, sparse_matrix):
      """Fit the model
      
      Fit the model starting at randomly initialized parameters. When
      warm_start=True, this method works the same as partial_fit.

      Parameters
      ----------
      
      sparse_matrix : sparse-matrix, shape (n_users, n_items)
         Sparse matrix in scipy.sparse format, can be created using sparse_matrix
         function from this package.
      """
      if not self.warm_start:
         self._fresh_session()
      return self.partial_fit(sparse_matrix)
   
   
   def partial_fit(self, sparse_matrix):
      """Fit the model
      
      Fit the model starting at previously trained parameter values. If the
      model was not trained yet, it randomly initializes parameters same as
      the fit method.

      Parameters
      ----------
      
      sparse_matrix : sparse-matrix, shape (n_users, n_items)
         Sparse matrix in scipy.sparse format, can be created using sparse_matrix
         function from this package.
      """
      if self._tf is None:
         self.init_with_shape(*sparse_matrix.shape)
      
      batch = self._batch_generator(sparse_matrix, size=self.batch_size,
                                    nonzero=not self.implicit)

      for _ in trange(self.n_iter, disable=not self.show_progress):
         batch_rows, batch_cols, batch_vals = next(batch)
         loss_value = self._tf.train(batch_rows, batch_cols, batch_vals)
         self.history.append(loss_value)
            
      return self
   
   
   def predict(self, rows, cols):
      """Predict using the model
      
      Parameters
      ----------
      
      rows : array, shape (n_samples,)
         Make predictions for those row indexes. If not provided,
         makes predictions for all the possible rows (use with caution).

      cols : array, shape (n_samples,)
         Make predictions for those  column indexes. If not provided,
         makes predictions for all the possible columns (use with caution).

      Returns
      -------
      array, shape (n_samples,)
         Predictions for given indexes.
      """
      return self._tf.predict(rows, cols)

   
   def predict_all(self, rows=None, cols=None):
      """Make predictions for the whole matrix, or slices of the matrix

      Parameters
      ----------
      
      rows : array, shape (n_samples,)
         Make predictions for those row indexes. If not provided,
         makes predictions for all the possible rows (use with caution).

      cols : array, shape (n_samples,)
         Make predictions for those  column indexes. If not provided,
         makes predictions for all the possible columns (use with caution).

      Returns
      -------
      scipy.sparse.csr_matrix, shape (n_rows, n_cols)
         Matrix of predictions for given indexes.
      """
      if cols is None and rows is None:
         rows = np.repeat([x for x in range(self.shape[0])], self.shape[1])
         cols = np.array([x for x in range(self.shape[1])] * self.shape[0])
      elif cols is None:
         cols = np.array([x for x in range(self.shape[1])] * len(rows))
         rows = np.repeat(rows, self.shape[1])
      elif rows is None:
         rows = np.array([x for x in range(self.shape[0])] * len(cols))
         cols = np.repeat(cols, self.shape[0])

      preds = self._tf.predict(rows, cols)

      return sparse_matrix(rows, cols, preds, shape=self.shape[:2], mode='csr')