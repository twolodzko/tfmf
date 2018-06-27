This package implements MatrixFactorizer class that can be used for
collaborative filtering using matrix factorization (see Koren et al, 2009,
Yu et al, 2012, Hu et al, 2008). It implements matrix factorization for
implicit and explicit ratings. The MatrixFactorizer class works with sparse
data and should be able to scale due to being programmed in TensorFlow.
The models can be easily saved for later use and restored.

.. code-block:: python

    import numpy as np
    from tfmf import MatrixFactorizer, sparse_matrix

    user_id = [0,0,1,1,2,2]
    movie_id = [0,1,2,0,1,2]
    rating = [1,1,2,2,3,3]
    X = sparse_matrix(user_id, movie_id, rating)

    mf = MatrixFactorizer(n_components=2, implicit=False)
    mf.fit(X)
    ## MatrixFactorizer(alpha=1.0, batch_size=500, fit_intercepts=True,
    ##        implicit=False, learning_rate=0.01, log_weights=None,
    ##        loss='squared', n_components=2, n_iter=500, optimizer='Adam',
    ##        random_state=None, regularization_rate=0.02, show_progress=True,
    ##        warm_start=False)
    
    X_full = np.array([[i,j] for i in range(3) for j in range(3)])
    np.reshape(mf.predict(X_full[:,0], X_full[:,1]), (3,3))
    ## array([[0.9903278, 1.0017678, 1.0006938],
    ##        [1.9346129, 2.021841 , 2.0176005],
    ##        [2.8033493, 2.9604366, 2.9534285]], dtype=float32)
    X.A
    ## array([[1, 1, 0],
    ##        [2, 0, 2],
    ##        [0, 3, 3]])

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
