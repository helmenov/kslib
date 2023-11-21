import numpy as np
from matplotlib import pyplot as plt

"""ICAの問題設定
観測信号 x_m(t) (m=1,...,M, t:time) は，原信号 s_n(t) (n=1,...,N, t:time) が混合したものである．

x_m(t) = \sum_n^{N} a_{m,n} * s_n(t)
       = \sum_n^{N} \sum_{\tau=0}^{T} a_{m,n}(\tau) * s_n(t-\tau)

- Tが0のとき，BSS: Blind Source Separation(ブラインド音源分離)
- T>0のとき，BSD: Blind Source Deconvolution(ブラインdの音源逆畳み込み)

# BSS

BSSのとき，tは無意味であり，各時刻の信号は単なる標本と考えてよい

s \in (N,) , x \in (M,) , A[m,n]=a_{m,n} とすると
x = A @ s
ここで，原信号sの推定値をy = \hat(s)とすると
y = W @ x, W \in R{n,m}

簡単のために，xは平均化した x = x-E[x]とする．

y = W@x = W@A@s なので，y=sである必要条件は，
y=s iff W@A=Eye \in {n,n}
"""

def ica_mle(x,ax):
    """MLEによる手法
    sの各チャネルの確率分布r(s)が既知とする．
    このとき，xの確率分布p(x)との関係は
    \int p(x) dx = \int r(s) ds

    dx = |\frac{\partial x_m}{\partial s_n}| ds
       = det(A) ds = |A|

    したがって，
    p(x) |A| = r(s)
    で，
    p(x) = r(s) |A|^{-1}
         = r(W@x)|W|

    xの対数尤度は，L = \sum_t^T log(p(x(t))) で

    L = \sum_t^T log(r(W@x(t)) |W| )
      = \sum_t^T log(r(W@x(t)) + T log(W)

    これを最大にする W を W* とすると
    (I) ... \frac{\partial L(W*)}{\partial W} = 0
    が必要条件

    1.
    \frac{\partial log|W|}{\partial W} = \frac{1}{W}\frac{\partial |W|}{\partial W}
    で，|W|の微分は W^{\top}の余因子行列 W0^{\top}であるので，
    \frac{\partial log|W|}{\partial W} = \frac{1}{W}W0^{\top} = (W^{\top})^{-1}

    2.
    \frac{\partial log(r(W@x))}{\partial W} = \frac{\partial log(r(y))}{\partial y} x^T

    2.1.
    r(y) = r(y_1) * r(y_2) * ... * r(y_N) なので，
    log r(y) = \sum_{n}^{N} log r(y_n)
    したがって，
    \frac{\partial log r(y)}{\partial y} = \sum_{n}^{N} \frac{\partial log r(y_n)}{\partial y}
    = \begin{pmatrix}
        \frac{d log r(y_1)}{dy_1} & \dots & \frac{d log r(y_N)}{dy_N}
      \end{pmatrix}^{\top}
    = \begin{pmatrix}
        \frac{r'(y_1)}{r(y_1)} &  \dots & \frac{r'(y_N)}{r(y_n)}
      \end{pmatrix}^{\top}
    = \phi(y)
    とする

    2.2.
    \frac{\partial log(r(W@x))}{\partial W} = \phi(y) x^{\top} = \phi(W@x) x^{\top}

    したがって，(I)は
    (I) ... \sum_t^T \phi(W@x(t)) x(t)^{\top} + T (W^{\top})^{-1} = 0

    しかし，これを解くことは困難
    """

def fit_minMI(x, eta, maxIter,yplot=True):
    """平均相互情報量最小化

    yの独立性尺度を満たすようにWを更新していく．

    独立性の尺度として「平均相互情報量(Mutual Information;MI)」がある．
    I(y) = \sum_{n}^{N} H({y_n}) - H({y_1, y_2, \dots, y_N}). H(y)はエントロピー
    yが独立なとき，y_nそれぞれのエントロピーの和が，y_n全体のエントロピーと等しい．
    I(y)が低いほど独立性が高い

    エントロピーH({y}) = - \int q(y) log q(y) dy

    x = A@s において， p(x) = r(s) |A|^{-1} であったように，
    y = W@x において，q(y) = p(x) |W|^{-1} である．すなわち
    q(y) = p(W^{-1}@y) |W|^{-1}

    したがって，
    H({y}) = - \int q(y) log q(y) dy
           = - \int p(W^{-1}@y)|W|^{-1} log{p(W^{-1}@y)|W|^{-1}} dy
    dyは，\frac{d}{dx}y = \frac{d}{dx}W@x = |W|より，dy = |W|dx なので
           = - \int { log p(x) - log |W| } p(x) |W|^{-1} |W| dx
           = - \int { log p(x) - log |W| } p(x) dx
           = log |W} - \int p(x) log p(x) dx
           = H({x}) + log |W|

    よって，平均相互情報量は，
    I(y) = \sum_{n}^{N} H(y_n) - H(y)
         = \sum_{n}^{N} H(y_n) - H(x) - log|W|

    これを最小化する．\frac{\partial}{\partial W}I_y(W) = 0

    \frac{\partial}{\partial W}I_y(w) = -(W^{\top})^{-1} - E[\phi(y)x^{\top}]
    \phi(y) = \frac{d}{dy}log q(y)

    これは，実はMLEと同じ．

    """
    from IPython.display import display
    M,L = x.shape
    # M: チャネル, L: span

    W = np.eye(M) # 単位ベクトルM本とする．
    x = normalize(x)

    ite=0
    if yplot==True:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,aspect='equal')
        Ln, = ax.plot(x[0,:],x[1,:],'r.')
        plt.ion()
    while ite < maxIter:
        y = W @ x
        #DeltaW = np.linalg.inv(W.T) + phi(y) @ x.T / L #最急勾配
        DeltaW = (np.eye(M) + phi(y) @ y.T / L) @ W  #自然勾配
        W = W + eta * DeltaW
        W = W/np.linalg.norm(W)
        if yplot==True:
            Ln.set_xdata(y[0,:])
            Ln.set_ydata(y[1,:])
            display(fig)
            plt.pause(1)
            #ax.cla()

    return W

def normalize(x,axis=1):
    mu = np.tile(x.mean(axis=axis),(x.shape[axis],1)).T
    x = x - mu
    s = np.diag((x**2).mean(axis=axis)**(1/2))
    x = s @ x
    return x

def phi(z):
    """本来未知である原信号の確率分布をp(z)としたときの，その対数微分
    最低限，各復元信号zの4次キュムラントの符号により切り替える必要がある．
    c4>0となる分布をSuper-Gaussian
    c4<0となる分布をSub-Gaussian
    とグループ分けしている．
    一様分布はc4<0なので，Sub-Gaussian
    正規分布の3乗は，c4>0なので，Super-Gaussianである．
    <Extended Informax>
    Super-Gaussianのとき，phi(x)=lambda x:-(x+tanh(x))
    Sub-Gaussianのとき，phi(x)=lambda x:-(x-tanh(x))
    すなわち，phi(x) = lambda x: -(x sign(c4)*tanh(x))
    """
    return -(z + np.diag(np.sign(cumulant(z,4))) @ np.tanh(z))
    # -np.tan(z)

def cumulant(x,d):
    if d==4:
        e4 = np.mean(x**4,axis=1)
        e2 = np.mean(x**2,axis=1)
        c4 = e4-3*e2**2
        return c4

def fit_with_preWhitening(x, eta, maxIter,yplot=True):
    M,L = x.shape
    # M: チャネル, L: span

    W = np.eye(M) # 単位ベクトルM本とする．

    #x = normalize(x)
    x, V, D = preWhitening(x)

    ite=0
    if yplot==True:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,aspect='equal')
        Ln, = ax.plot(x[0,:],x[1,:],'r.')
        plt.ion()
    while ite < maxIter:
        y = W @ x
        DeltaW = ((phi(y) @ y.T - y @ phi(y).T)/ L) @ W
        W = W + eta * DeltaW
        if np.linalg.norm(DeltaW) < 1e-7:
            break
        if yplot==True:
            Ln.set_xdata(y[0,:])
            Ln.set_ydata(y[1,:])
            display(fig)
            plt.pause(1)
            #ax.cla()

    return W

def preWhitening(x):
    L = x.shape[1]
    x = x - np.tile(np.mean(x,axis=1),(L,1)).T
    D, V = np.linalg.eig(x @ x.T /L)
    D = 1/np.sqrt(D)
    F = V @ np.diag(D) @ V.T
    print(F.shape)
    return F @ x, V, D

