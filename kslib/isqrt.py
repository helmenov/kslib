import numpy as np

def first_inverse_sqrt(n):
    """
    return n^{-1/2} the minus square root, but not inverse of sqrt (n^2)
    """
    threehalfs = 1.5
    x2 = n * 0.5
    y = np.float32(n)

    i = y.view(np.int32)
    i = np.int32(0x5f3759df) - np.int32(i >> 1)
    y = i.view(np.float32)

    y = y * (threehalfs - (x2 * y * y))
    return float(y)

def sqrt(n):
    newton_constant = 0.5
    x2 = n * 0.5
    y  = np.float32(n)

    i = y.view(np.int32)
    i = np.int32(0x1FBD1E2D) + np.int32(i >> 1)
    y = i.view(np.float32)

    y = y * newton_constant + x2 / y
    # y = y * newton_constant + x2 / y # 2nd loop

    return float(y)

