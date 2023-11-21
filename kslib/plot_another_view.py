from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

"ax0の3dプロットをax1にコピーする．ついでに視点をずらす．"
def plot_AnotherAxesWithAnotherview(Axes:plt.Axes, srcAxes:plt.Axes):
    """
    fig, nrows:int,ncols:int,index:int)->list[plt.Axes]:
    ax = list()
    _i -= 1
    _r = _i // ncols
    _c = _i % ncols
    r = _r
    cL = int(_c * 2)
    ncols = 2*ncols
    indexL = int(r*ncols+cL+1)
    indexR = int(indexL+1)
    ax[0] = fig.add_subplot(nrows,ncols,indexL,projection='3d')
    ax[1] = fig.add_subplot(nrows,ncols,indexR,projection='3d')
    ax[0].view_init(elev=25, azim=-60)
    ax[1].view_init(elev=35, azim=-60)
    """
    fig = plt.figure()
    ax = fig.add_subplot(3,2,1,projection='3d')
    ax.plot(range(10))
    ax2 = fig.add_subplot(3,2,2,projection='3d')
    ax2.update_from(ax)



