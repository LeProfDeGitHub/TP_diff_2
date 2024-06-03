from __future__ import annotations
import os
import shutil
import time
from typing import Callable, Concatenate

import numpy as np



FOLDER_ALREADY_INIT = False



def init_figure_folder():
    """
    Create a new folder 'figure' if they don't already exist, else delets and recreats it.
    If the folder is already initialized, it does nothing.
    """
    global FOLDER_ALREADY_INIT

    if os.path.exists('figure') and not FOLDER_ALREADY_INIT:
        shutil.rmtree('figure')
        FOLDER_ALREADY_INIT = True
    else:
        print('The folder is already initialized.')

def add_floders(floders: tuple[str, ...]):
    """
    Add subfolders to the 'figure' folder. 
    """
    os.makedirs('figure')
    for floder in floders:
        os.makedirs(f'figure\\{floder}')

def format_path(label: str) -> str:
    """
    Format the path to the figure by adding the 'figure' folder.
    """
    path = label.lower().replace(' ', '_')
    return path


class TestFuncsCollection:
    '''
    A decorator to print the current number of the figure over the total number of figures.
    '''
    def __init__(self, label: str) -> None:
        '''
        Initialize the current number of the figure and the total number of figures.
        '''
        self.current_nbr = 0
        self.total_nbr = 0
        self.funcs = []
        self.label = label

    def print_current_nbr(self):
        '''
        Print the current number of the figure over the total number of figures.
        '''
        nbr_carac = len(str(self.total_nbr))
        print(f'{self.label} >> {self.current_nbr:>{nbr_carac}}/{self.total_nbr} : ', end='')

    def __call__[**P](self, func: Callable[Concatenate[TestFuncsCollection, P], None], n: int = 1) -> Callable[P, None]:
        '''
        Decorate the function to print the current number of the figure over the
        total number of figures. The number of figures is incremented by n.
        '''
        self.total_nbr += n
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
            func(self, *args, **kwargs)
        self.funcs.append(wrapper)
        return wrapper
    

def test_deco_n(test_funcs_collection: TestFuncsCollection, n: int):
    '''
    A decorator to print the current number of the figure over the total number of figures.
    '''
    def decorator[**P](func: Callable[Concatenate[TestFuncsCollection, P], None]) -> Callable[P, None]:
        '''
        Decorate the function to print the current number of the figure over the
        total number of figures. The number of figures is incremented by n.
        '''
        return test_funcs_collection(func, n)
    return decorator

def time_func[**P, R](func: Callable[P, R]) -> Callable[P, tuple[float, R]]:
    '''
    Return the time taken to execute the function.
    '''
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[float, R]:
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        return end - start, res
    return wrapper

def log_range(start: float, stop: float, num: int, base: float = np.e) -> np.ndarray:
    '''
    Renvoie une plage de valeur avec une distribution logarithmique (concentré sur le début et dissipé vers la fin).
    '''
    log_range: np.ndarray = np.logspace(np.log(start), np.log(stop), num, base=base, endpoint=True)
    return log_range

