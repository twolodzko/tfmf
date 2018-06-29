This package implements ``MatrixFactorizer`` class for collaborative filtering
using matrix factorization with implicit (Hu et al, 2008) and explicit ratings
(Koren et al, 2009, Yu et al, 2012). The ``MatrixFactorizer`` class works with
sparse data (scipy.sparse_ matrices) and should be able to scale due to using
TensorFlow_ backend. The trained models can be easily saved and restored.

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
and random batches of size 4 for the training. To make predictions from the model, we
use ``predict_all`` method that makes predictions for the whole factorized matrix
(or slices of it), but more standard choice would be to use ``predict(user_ids, item_ids)``
method for making predictions for individual users and items.

.. code-block:: python

    mf = MatrixFactorizer(n_components=2, batch_size=4, implicit=False)
    mf.fit(X)
    mf.predict_all().A
    ## array([[1.013387 , 1.0056534, 1.11131  ],
    ##        [1.926976 , 1.8363876, 1.9962051],
    ##        [1.9263643, 2.969007 , 2.9673567]], dtype=float32)
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
    mf.predict_all().A
    ## array([[ 4.6303128e-05, -5.7351419e-05,  1.7567796e-05],
    ##        [-1.0485943e-04,  6.2389910e-05, -3.7540267e-05],
    ##        [-1.2580390e-04, -2.4326339e-04, -3.4460012e-05]], dtype=float32)

Instead of training, we resore the previously learned parameters. Such restored model
can be trained further using the ``partial_fit`` method, or used to make predictions.

.. code-block:: python

    mf.restore(tmpdir + '/tfmf')
    mf.predict_all().A
    ## array([[1.013387 , 1.0056534, 1.11131  ],
    ##        [1.926976 , 1.8363876, 1.9962051],
    ##        [1.9263643, 2.969007 , 2.9673567]], dtype=float32)

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


.. _scipy.sparse: https://docs.scipy.org/doc/scipy/reference/sparse.html
.. _TensorFlow: http://tensorflow.org/