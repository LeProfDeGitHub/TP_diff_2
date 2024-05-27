import os
import shutil
from typing import Callable





def add_floders(floders: tuple[str, ...]):
    """
    Create a new folder 'figure' and subfolders 'grad_desc_fix_step' and 'grad_desc_optimal_step'
    if they already exist, they are deleted and recreated.
    """
    if os.path.exists('figure'):
        shutil.rmtree('figure')

    os.makedirs('figure')
    for floder in floders:
        os.makedirs(f'figure\\{floder}')

def print_current_nbr(current_nbr: int, total_nbr: int):
    '''
    Print the current number of the figure over the total number of figures.
    '''
    nbr_carac = len(str(total_nbr))
    print(f'{current_nbr:> {nbr_carac}}/{total_nbr} : ', end='')

class DisplayFunc:
    '''
    A decorator to print the current number of the figure over the total number of figures.
    '''
    def __init__(self) -> None:
        '''
        Initialize the current number of the figure and the total number of figures.
        '''
        self.current_nbr = 0
        self.total_nbr = 0
        self.funcs = []
    
    def __call__(self, func: Callable, *args, n: int = 1, **kwargs):
        '''
        Decorate the function to print the current number of the figure over the
        total number of figures. The number of figures is incremented by n.
        '''
        self.total_nbr += n
        def wrapper():
            self.current_nbr += n
            print_current_nbr(self.current_nbr, self.total_nbr)
            func(*args, **kwargs)
        self.funcs.append(wrapper)
        return wrapper
    
display_func = DisplayFunc()

def display_func_n(n: int):
    '''
    A decorator to print the current number of the figure over the total number of figures.
    '''
    def decorator(func: Callable, *args, **kwargs):
        '''
        Decorate the function to print the current number of the figure over the
        total number of figures. The number of figures is incremented by n.
        '''
        return display_func(func, *args, n=n, **kwargs)
    return decorator


