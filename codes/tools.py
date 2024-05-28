from __future__ import annotations
import os
import shutil
from typing import Callable, Concatenate





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
