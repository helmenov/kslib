#%%
import numpy as np

def YearToRetire(S, r):
    """
    - 貯蓄が支出25年分あればRetireできる．
    - 貯蓄を投資して4%切り崩すと1年分の支出
    

    S: 年間貯蓄率 (0,100)
    r: 年間投資リターン (0,100)
    """
    n = np.log(25 * (1-S)/S * r +1)/np.log(r+1)
    Y = int(n)
    M = (n-Y)//12
    return (Y,M)

def SaveToRetire(Y,r):
    S = 1/((np.exp(Y*np.log(r+1))-1)/(25*r)+1)
    return S
#%%
S = 0.1
r = 0.02
(Y,M ) = YearToRetire(S,r)
print(f'FIREまで{Y}年{M}ヶ月')
S2 = SaveToRetire(Y,r)
print(f'貯蓄率は{S2:3.1f}')
# %%
