
This package implements MatrixFactorizer class that can be used for matrix
factorization using stochastic gradient descent (see Koren et al, 2009).
It aims at providing simple interface and enabling user to fit the model
in online fashion.

The data is assumed to come in the long format as below:

.. code-block:: python

    user_id movie_id rating
    1471    231      1
    1233    27       5
    1471    4566     2
    ...     ...      ...

what is equivalent to storing the ratings in the user_id * movie_id matrix. 

The MatrixFactorizer class uses interface similar to the classes in the
sklearn package. The typical workflow includes pre-processing, cleaning, etc.
and then spliting the data into arrays X for indexes and y for the values,
after that, the MatrixFactorizer model an be fitted.

.. code-block:: python

    from sgdmf import MatrixFactorizer

    X = data[['user_id', 'movie_id']]
    y = data['rating']

    mf = MatrixFactorizer()
    mf.partial_fit(X, y)

Since MatrixFactorizer is fitted using stochastic gradient descent,
it is possible to train model using online learning in batches
(or case by case), using the partial_fit() function. MatrixFactorizer
is able to adapt to changes in the data (new indexes) by initializing
the parameters for yet unseen indexes on-the-fly (when using
dynamic_indexes=True).

References
----------
               
Koren, Y., Bell, R., & Volinsky, C. (2009).
Matrix factorization techniques for recommender systems. Computer, 42(8).
