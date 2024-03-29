# KS Lib

Author: Kotaro Sonoda

## Libraries


- `beta/` : 作りかけ
    - `plot_another_view.py`: 視点をずらした3Dプロットを並べてステレオ立体視させるもの
    - `diviser.py` :
        - `factorization(n:int)->List[Tuple[int,int]`: 自然数nの因数分解．ex) 60 -> [(2,2),(3,1),(5,1)] := 2^2 * 3^1 * 5^1
        - `divisors(n:int)->List[int]` : 自然数nの約数のリスト．
        60 -> [1, 2, 3, 4, 5, 6, 10, 12, 15, 30]
        - `num_of_divisor(n: int)->int` : 自然数nの約数の数．ex) 60 -> 10 := len([1, 2, 3, 4, 5, 6, 10, 12, 15, 30])
        - `sum_of_divisors(n:int)->int` : 自然数nの約数の和．ex)
        60 -> 88 := sum([1, 2, 3, 4, 5, 6, 10, 12, 15, 30])
        - `sum_of_inv_divisors(n:int)` : 自然数nの約数の逆数の和
        60 ->  : 1/1 + 1/2 + ... + 1/30
        - `is_perfectnum(n:int)->Bool` : 自然数nは完全数か
        - `perfectnums_leq(n)->List[int]` : 自然数n以下の完全数リスト
        - `perfectnums(n)->List[int]` : 2から数えてn個の完全数リスト
        - `ret_sum_of_inv_divisors_is(p,q)->int`: 約数の逆数の和がp/qとなる整数を求める
    - `reduct_frac(p:float,q:int)->Tuple[int,int]` : (float)p/(int)q の有理化
    - `class fourieranalysis(x)` :
        - Freq domains
            - X, X_min, X_all
            - amp, amp_min, amp_all, dB
            - phase, phase_min, phase_all, rcphase
        - Quef domains
            - C_complex, C_min, C_all
            - C_amp, C_amp_min, C_amp_all
            - C_phase, C_phase_min, C_phase_all
        - Time domains
            - xmin, xall
        - Notes
            - *min 最小位相, *all オールパス
            - 各ドメインで，無印=最小位相+オールパス
            - 各ドメインで，無印=amp * exp(j phase)

## 方針

0. 基盤となる関数をいれる．
    - プロジェクトのパッケージでimportされるもの
1. とりあえずは

