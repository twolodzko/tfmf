This package implements ``MatrixFactorizer`` class for collaborative filtering
using matrix factorization with implicit (Hu et al, 2008) and explicit ratings
(Koren et al, 2009, Yu et al, 2012). The ``MatrixFactorizer`` class works with
sparse data (``scipy.sparse`` matrices) and should be able to scale due using
TensorFlow backend. The trained models can be easily saved and restored.

Example
-------

Below an example of using ``MatrixFactorizer`` class for explicit ratings is shown.
The data is converted to to ``scipy.sparse.dok_matrix`` format using the
``sparse_matrix`` function provided in the package as a convenience wraper.

.. code-block:: python

    import numpy as np
    from tfmf import MatrixFactorizer, sparse_matrix

    user_id = [0,0,1,1,2,2]
    movie_id = [0,1,2,0,1,2]
    rating = [1,1,2,2,3,3]
    X = sparse_matrix(user_id, movie_id, rating)

The model is trained using the ``fit`` method. We choose to use two latent dimmensions
and random batches of size 4 for the training. 

.. code-block:: python

    mf = MatrixFactorizer(n_components=2, batch_size=4, implicit=False)
    mf.fit(X)
    X_full = np.array([[i,j] for i in range(3) for j in range(3)])
    np.reshape(mf.predict(X_full[:,0], X_full[:,1]), (3,3))
    ## array([[1.027492  , 0.99959517, 1.0010808 ],
    ##        [1.9425895 , 2.033092  , 2.0300255 ],
    ##        [2.738347  , 2.9628766 , 2.954407  ]], dtype=float32)
    X.A
    ## array([[1, 1, 0],
    ##        [2, 0, 2],
    ##        [0, 3, 3]])

The learned parameters are saved to temporary directory.

.. code-block:: python

    import tempfile
    tmpdir = tempfile.gettempdir()
    mf.save(tmpdir + '/tfmf')

Next, we initialize the class once again. Using the ``init_with_shape`` method,
``MatrixFactorizer`` class is set-up to model the matrix of shape (n_users, n_items)
with randomly initialized parameters. Predictions from such model are meaningless,
because it wasn't trained yet.

.. code-block:: python

    mf = MatrixFactorizer(n_components=2, batch_size=4, implicit=False)
    mf.init_with_shape(3, 3)
    np.reshape(mf.predict(X_full[:,0], X_full[:,1]), (3,3))
    ## array([[-3.88692170e-05,  1.02325015e-04,  3.66907625e-04],
    ##        [-3.07143564e-05, -2.27056171e-05,  2.18227942e-05],
    ##        [-2.47780608e-05, -1.64722569e-05,  2.23812822e-05]], dtype=float32)

Instead of training, we resore the previously learned parameters. Such restored model
can be trained further using the ``partial_fit`` method, or used to make predictions.

.. code-block:: python

    mf.restore(tmpdir + '/tfmf')
    np.reshape(mf.predict(X_full[:,0], X_full[:,1]), (3,3))
    ## array([[1.027492  , 0.99959517, 1.0010808 ],
    ##        [1.9425895 , 2.033092  , 2.0300255 ],
    ##        [2.738347  , 2.9628766 , 2.954407  ]], dtype=float32)

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
