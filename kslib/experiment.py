from pathlib import Path
import string
from datetime import datetime
import sys
import pickle

def get_rnd_suffix() -> str:
    return '_' + ''.join(random.choices(string.ascii_letters + string.digits, k=8))

def get_date_suffix() -> str:
    return '_' + datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")

def make_result_dir(prefix:str,suffix_type:str) -> Path:
    prefix = Path(prefix)
    prefix_parent = prefix.parent
    prefix_name = prefix.name
    if prefix_parent.is_dir():
        pass
    else:
        prefix_parent.mkdir(exist_ok=False)
    suffix_func = {'random':get_rnd_suffix, 'date':get_date_suffix}
    res_dir = prefix_parent.joinpath(prefix_name + suffix_func['date']())
    res_dir.mkdir(exist_ok=False)
    return res_dir

def logger(prefix_dir):
    """Decorator 'logger'
    @logger(prefix_dir)
    def func1(x,y):
        z = x+y
        retunr z

    z = func1(2,3)

    is logging to res_dir/command.log that

    func1 starts...
    ====args====
    2
    3
    """
    prefix_dir = Path(prefix_dir)

    def _logger(Func):
        def wrapper(*args, **keywords):
            res_dir = make_result_dir(prefix_dir,'date')
            with res_dir.joinpath('command.log').open('a') as f:
                f.writelines(f'{Func.__name__} starts...\n')
                f.writelines('\n')
                f.writelines('====args and keywords====\n')
                for i,a in enumerate(args):
                    f.writelines(f'arg[{i}]:{a}\n')
                for k in keywords:
                    f.writelines(f'{k}:{keywords[k]}\n')
                f.writelines('\n')
                f.writelines('====date====\n')
                date_begin = datetime.now()
                f.writelines(f'begin:\t{date_begin}\n')
                v = Func(*args, **keywords)
                date_end = datetime.now()
                f.writelines(f'end:\t{date_end}\n')
                date_past = date_end - date_begin
                f.writelines(f'Past Time:\t{date_past}\n')
                f.writelines('\n')
                f.writelines('====Result===\n')
                if isinstance(v,tuple):
                    for i,vi in enumerate(v):
                        f.writelines(f'<<out[{i}]>>\n')
                        f.writelines(f'{vi}\n')
                else:
                    f.writelines('<<out>>\n')
                    f.writelines(f'{v}\n')
                with res_dir.joinpath('results.pickle').open('wb') as p:
                    pickle.dump(v,p)
                return v
        return wrapper
    return _logger

