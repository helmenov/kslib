from kslib import reduct_frac

def factorization(n: int)->List[Tuple[int,int]]:
    """Factorize Natural number

    Args:
        n (int): Natural number

    Returns:
        List[Tuple[int,int]]: Factor list of tuple(prime,power)
    """
    fact = []
    temp = n
    for i in range(2, int(-(-n**0.5//1))+1):
        if temp%i==0:
            cnt=0
            while temp%i==0:
                cnt+=1
                temp //= i
            fact.append((i, cnt))

    if temp!=1:
        fact.append((temp, 1))

    if fact==[]:
        fact.append((n, 1))

    return fact

def num_of_divisor(n: int)->int:
    """Count of Divisor of Natural number

    Args:
        n (int): Natural Number

    Returns:
        int: Count
    """
    a = 1
    for _, x in factorization(n):
        a *= x + 1
    return a

def sum_of_p_divisors(p,n):
    a = 0
    while n > 0:
        a += p ** n
        n -= 1
    return a + 1

def sum_of_divisors(n:int)->int:
    a = 1
    for p, q in factorization(n):
        a *= sum_of_p_divisors(p,q)
    return a

def sum_of_inv_divisors(n):
    return reduct_frac.reduct_frac(sum_of_divisors(n),n)

def is_perfectnum(n:int)->bool:
    """Check n is perfect number or not

    Args:
        n (int): Natural Number

    Returns:
        bool: Result
    """
    if sum_of_divisors(n) == 2*n:
        return True
    else:
        return False

def p_divisors(p,n):
    a = []
    for i in range(0, n + 1):
        a.append(p ** i)
    return a

def divisors(n:int)->List[int]:
    xs = factorization(n)
    ys = p_divisors(xs[0][0], xs[0][1])
    for p, q in xs[1:]:
        ys = [x * y for x in p_divisors(p,q) for y in ys]
    return sorted(ys)

def perfectnums_leq(n:int)->List[int]:
    """List-up perfect numbers less than or equal n

    Args:
        n (int): Natural number

    Returns:
        List[int]: perfect numbers
    """
    p = []
    for x in range(2,n+1):
        if is_perfectnum(x):
            p.append(x)
    return p

def perfectnums(n:int)->List[int]:
    """Return list of n perfect numbers

    Args:
        n (int): wanted size of list

    Returns:
        List[int]: list of perfect numbers
    """
    x = 2
    p = []
    while len(p) < n:
        if is_perfectnum(x):
            print(x)
            p.append(x)
        x += 1
    return p

def ret_sum_of_inv_divisors_is(p,q):
    ret = []
    for i in range(1,100000):
        p1 = i*p
        q1 = i*q
        g = sum_of_divisors(q1)
        if g == p1:
            ret.append(q1)
    return ret




# 約数の逆数の和がSとなる整数を求める
# https://youtu.be/tLKwMzpRl8w
