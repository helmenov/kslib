import numpy as np

def SVD(X, rank):
    """Singular Value Decomposition

    Args:
        X : target matrix, Shape:(M,N)

    Returns:
        s : Eigen values, Shape:(D,)
        U : Left Eigen Vectors, Shape:(M,D)
        V : Right Eigen Vectors, Shape:(N,D)

    Definition X@V = U@S
    """
    M, N = X.shape
    rg = np.random.default_rng()
    Omega = rg.standard_normal((N,k))
    Y = X @ Omega
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ X
    Uhat, s, V = np.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat

    return s, U, V


def SSA(x,width):
    """Singular Spectrum Analysis"""
    len_x = len(x)
    X = np.array([x[i:i+width] for i in range(len_x-width)]).T



