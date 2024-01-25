import numpy as np

def Otsu(x):

    def VarWithin(x,t):
        y = np.full_like(x,False)
        y[x>=t] = True
        N = len(x)
        n1 = np.count_nonzero(y)
        w1 = n1/N
        w0 = 1-w1

        if w0 == 0 or w1 == 0:
            return np.nan

        y1 = x[y==True]
        y0 = x[y==False]

        v0 = np.var(y0) if len(y0)>0 else 0
        v1 = np.var(y1) if len(y1)>0 else 0

        return w0 * v0 + w1 * v1

    dt = min(np.diff(sorted(x)))
    t_min = min(x)
    t_max = max(x)
    t_range = np.arange(t_min,t_max,dt)
    criteria_ = [VarWithin(x,t) for t in t_range]
    t = t_range[np.argmin(criteria_)] # best_threshold

    y = np.array([True if xi > t else False for xi in x])

    return y
