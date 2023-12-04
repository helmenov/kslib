import numpy as np
from Typing import Tuple

def fraction_expand(p:int, q:int)->Tuple[int,int,int]:
    """p/qを整数部n0と小数部に分け，小数部を1/(p0/q0)にする

    Args:
        p (float): numerator
        q (int): denominator
    Output:
        n0 (int): 整数部
        p0 (int): 小数部を1/(p0/q0)で表したときのp0
        q0 (int): 小数部を1/(p0/10)で表したときのq0
    """
    if p!= int(p):
        n0 = int(np.floor(p/q))
        r = p/q-n0
        p0 = 1/r
        q0 = int(1)
    else:
        n0 = int(p//q)
        p0 = q
        q0 = int(p-n0*q)
    return (n0,p0,q0)

def fraction_shrink(n:int,p:int,q:int)->Tuple[int,int]:
    """ n + 1/(p/q) -> p0/q0

    Args:
        n (int): _description_
        p (int): _description_
        q (int): _description_

    Returns:
        Tuple[int,int]: _description_
    """
    p0 = n*p+q
    q0 = p
    return (p0,q0)

def reduct_frac(p:float,q:int)->Tuple[int,int]:
    """
    p/qを約分したa/bを返す．
    具体的には，$p/q$を
    1. $p_0/q_0$を整数部$N_0$と小数部$N_1$に分離し，
        $N_0 = p_0//q_0, N_1 = p_0\%q_0$
        $p_0$がfloatであっても上記の方法で整数部と小数部を分離できる．
    2. 小数部$N_1$を\frac{1}{\frac{1}{N_1}}$に直す．
    3. $p_1 = \frac{1}{N_1}$を計算し，p_1,q_1=1を上記の１へ
    4. 1〜4を繰り返し，$(N_k-1)^2 < \text{eps}$で繰り返しを抜ける．
    5. N_k = 1として，逆にたどっていく． N_{k-1} =

    Args:
        p (float): 分子
        q (int): 分母
    Output:
        (Tuple(int,int)): 分子と分母を返却する
    """
    if q != int(q):
        p1,q1 = reduct_frac(q,1)
        q0 = p1
        p0 = p*q1
    else:
        p0 = p
        q0 = q

    n = []
    while q0 > 0:
        nn, p0, q0 = fraction_expand(p0,q0)
        # print(f'p/q => {nn} + 1/({p0}/{q0})')
        n.append(nn)
        if len(n)>1 and n[-1] > 10:
            break
    p0 = n[-1]
    q0 = 1
    for i in range(len(n)-1):
        #print(f'p/q <= {n[-(i+2)]} + {q0}/{p0}')
        p0, q0 = fraction_shrink(n[-(i+2)],p0,q0)

    p = p0
    q = q0

    return p,q
