
import numpy as np
from tfmf import MatrixFactorizer, sparse_matrix, rank, top_k_ranks


if __name__ == "__main__":

    # test for obvious fails during initialization

    X = sparse_matrix([1,1,2,2,3,3], [1,2,3,0,1,3], [1,2,1,1,3,2])

    for test_implicit in [True, False]:
        for test_loss in ['squared', 'logistic']:
            for test_log_weights in [True, False]:
                for test_fit_intercepts in [True, False]:
                    for test_optimizer in ['Adam', 'Ftrl']:
                    
                        settings = {
                            'implicit' : test_implicit,
                            'loss' : test_loss,
                            'log_weights' : test_log_weights,
                            'fit_intercepts' : test_fit_intercepts,
                            'optimizer' : test_optimizer
                        }

                        print('Model: ', settings)

                        model = MatrixFactorizer(n_components=2, n_iter=10, batch_size=4,
                                                learning_rate=1, alpha=1,
                                                regularization_rate=0,
                                                random_state=42, warm_start=False,
                                                show_progress=False, **settings)

                        model.fit(X)
                        model.partial_fit(X)
                        model.predict([0,1], [0,1])


    # unit tests for rank and top_k_ranks functions

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