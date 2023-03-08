import numpy as np

def distort_clip(x,g,level):
    y = np.full_like(x)
    gx = g*x
    y[np.where(gx>=1)[0]] = level
    y[np.where(gx<=-1)[0]] = -level
    for i in np.where(np.logical_and(-1<gx,gx<1))[0]:
        y[i] = level*gx[i]

    return y

def distort_soft_clip(x,g,level,async=(1,1)):
    y = np.full_like(x)
    gx = g*x
    for i in np.where(gx>=0)[0]:
        y[i] = np.atan(gx[i])/(np.pi/2)*async[0]
    for i in np.where(gx<0)[0]:
        y[i] = np.atan(gx[i])/(np.pi/2)*async[1]
    y *= level

    return y

def distort_rectify(x,g):
    gx = g*x
    y = gx
    y[np.where(gx<0)[0]] *= -1
    return y

def limit(x,th):
    y = x
    for i in np.where(np.abs(x)>th)[0]:
        y[i] = th*np.sign(x[i])
    return y

def compress(x,th,ratio):
    gain = 1.0/(th+(1-th)*ratio)
    y = x
    for i in np.where(y > th)[0]:
        y[i] = th + (y[i]-th)*ratio
    for i in np.where(y < -th)[0]:
        y[i] = -th +(y[i]+th)*ratio
    y *= gain
    return y
