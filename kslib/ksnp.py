import numpy as np

def cofactor(A:np.ndarray)->np.ndarray:
    """Cofactor Matrix

    Args:
        A (np.ndarray): target matrix

    Returns:
        np.ndarray: cofactor matrix
    """
    def cofactor_ij(A:np.ndarray,i:int,j:int)->float:
        """Cofactor(余因子)
        m_ij : i-th row と j-th column を除いた行列の行列式
        のとき，
        余因子は，(-1)^{i+j} m_ij

        Args:
            A (np.ndarray): target matrix
            i (int): row number to delete
            j (int): col. number to delete

        Returns:
            (float): cofactor of minor_determinant_of(i,j)
        """
        _A = np.delete(A,i,axis=0)
        _A = np.delete(_A,j,axis=1)
        return (-1)**(i+j) * np.linalg.det(_A)
    R,C = A.shape
    Ac = np.zeros_like(A.T)
    for r in range(R):
        for c in range(C):
            Ac[c,r] = cofactor_ij(A,r,c)
    return Ac

def vec(x:np.ndarray,alpha:list[int]=None)->np.ndarray:
    """matrix flattened vector

    Args:
        x (np.ndarray): target matrix
        alpha (list[int], optional): row number's list to get. if None, whole rows. Defaults to None.

    Returns:
        np.ndarray: matrix flattened vector
    """
    x = np.array(x)
    M,N = x.shape
    e_n = stdBasisVector(0,N)
    ret = np.kron(e_n,x.dot(e_n))
    for n in range(1,N):
        e_n = stdBasisVector(n,N)
        ret += np.kron(e_n, x.dot(e_n))
    if alpha==None:
        return ret
    else:
        E = IndexMatrix(alpha,M)
        return vec(E.dot(x))

def stdBasisVector(n:int,N:int)->np.ndarray:
    """standard basis vector

    Args:
        n (int): the order of standard basis vectors
        N (int): rank of basis

    Returns:
        np.ndarray: n-th standard basis vector
    """
    return np.eye(N)[n]

def IndexMatrix(alpha:list[int],M:int)->np.ndarray:
    """Matrix to return alpha-th rows

    Args:
        alpha (list[int]): list of row-indices
        M (int): sup. of size of alpha

    Returns:
        (np.ndarray): index matrix (shape=(M,M))
    """
    d = len(alpha)
    assert 0 <= d <= M
    E = np.zeros(shape=(d,M))
    for i, a in enumerate(alpha):
        E[i,:] = stdBasisVector(a,M)
    return E

def alpha_c(alpha:list[int],M:int)->list[int]:
    """complement for the list of index_numbers of the size M array

    Args:
        alpha (list[int]): the list of index_numbers of the Size M array
        M (int): sup. of size of alpha

    Returns:
        list[int]: list of numbers of complement for alpha
    """
    a_c = list(range(M))
    for a in alpha:
        a_c.remove(a)
    return a_c

