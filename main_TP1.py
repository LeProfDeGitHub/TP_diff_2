from matplotlib import pyplot as plt
import numpy as np
from function import QuadraticFunction, get_other_diago, get_zvankin_quad, condi_A
from display import (display_convergence_2d,
                     display_convergence_by_X0,
                     display_partial_func,
                     display_norm,
                     display_error,
                     display_compare_error,
                     display_ka)
from methods import (gradient_descent_fix_step,
                     gradient_descent_optimal_step,
                     quadratic_gradient_descent,
                     quadratic_conjuguate_gradient_method,)
from tools import (add_floders,
                   display_func,
                   display_func_n,)

METHODS_PATH = ((gradient_descent_fix_step           , 'grad_desc_fix_step'    ),
                (quadratic_gradient_descent          , 'grad_desc_optimal_step'),
                (quadratic_conjuguate_gradient_method, 'conjuguate_gradient'   ),)

METHODS_LABELS = ((gradient_descent_fix_step           , 'Gradient Descent Fix Step'    ),
                  (quadratic_gradient_descent          , 'Gradient Descent Optimal Step'),
                  (quadratic_conjuguate_gradient_method, 'Conjuguate Gradient'          ),)

@display_func_n(3)
def display_all_converence_2d():
    '''
    Call display_convergence_2d for each method in METHODS_PATH.
    '''
    for method, path in METHODS_PATH:
        display_convergence_2d(f'figure\\{path}',
                               get_zvankin_quad(2),
                               np.array([[-5], [-5]]),
                               method)

@display_func_n(3)
def display_all_converence_by_X0():
    '''
    Call display_convergence_by_X0 for each method in METHODS_PATH.
    '''
    for method, path in METHODS_PATH:
        display_convergence_by_X0(f'figure\\{path}',
                                  get_zvankin_quad(2),
                                  (-10, 10),
                                  (-10, 10),
                                  50, 5e-2, 10000,
                                  method)

@display_func_n(3)
def display_all_partial_func():
    '''
    Call display_partial_func for each method in METHODS_PATH.
    '''
    for method, path in METHODS_PATH:
        display_partial_func(f'figure\\{path}\\partial_func',
                             get_zvankin_quad(2),
                             np.array([[-5], [-5]]),
                             method)

@display_func_n(3)
def display_all_norm():
    '''
    Call display_norm for each method in METHODS_PATH.
    '''
    for method, path in METHODS_PATH:
        display_norm(f'figure\\{path}',
                     get_zvankin_quad(2),
                     np.array([[-5], [-5]]),
                     method)

@display_func_n(3)
def display_all_error():
    '''
    Call display_error for each method in METHODS_PATH.
    '''
    for method, path in METHODS_PATH:
        display_error(f'figure\\{path}',
                      get_zvankin_quad(2),
                      np.array([[-5], [-5]]),
                      method)

@display_func_n(3)
def display_all_compare_error():
    '''
    Call display_compare_error for each method in METHODS_LABELS.
    '''
    display_compare_error(f'figure\\',
                          get_zvankin_quad(2),
                          np.array([[-5], [-5]]),
                          METHODS_LABELS)

@display_func_n(1)
def display_ka_wrp():
    '''
    Call display_ka for the condition number of the matrix A.
    '''
    display_ka('figure\\condition_number.png', 500)


def comparaison_condi(f : QuadraticFunction, X0, eps: float, niter: int):
    """
    compare the number of iterations of the gradient descent method
    and the conjugate gradient method for a given quadratic function.

    :param f: QuadraticFunction object
    :param X0: starting point
    :param eps: error
    :param niter: number of iterations
    :return: none
    """
    print("------------------------------------------")
    print("           unconditioned matrix           ")
    print("------------------------------------------\n")
    X_gd = quadratic_gradient_descent(f, X0, eps, niter)
    print(f'gradient descent method optimal step: {len(X_gd)} iterations')
    error = np.linalg.norm(f.df (X_gd[-1]))
    print(f'error: {error}\n')

    X_cg = quadratic_conjuguate_gradient_method(f, X0, eps, niter)
    print(f'conjuguate gradient method: {len(X_cg)} iterations')
    error = np.linalg.norm(f.df (X_cg[-1]))
    print(f'error: {error }\n')

    print("------------------------------------------")
    print("             conditioned matrix           ")
    print("------------------------------------------\n")

    quad = condi_A(f)
    X_gd = quadratic_gradient_descent(quad, X0, eps, niter)
    print(f'gradient descent method optimal step: {len(X_gd)} iterations')
    error = np.linalg.norm(quad.df (X_gd[-1]))
    print(f'error: {error }\n')

    X_cg = quadratic_conjuguate_gradient_method(quad, X0, eps, niter)
    print(f'conjuguate gradient method: {len(X_cg)} iterations')
    error = np.linalg.norm(quad.df (X_cg[-1]))
    print(f'error: {error}\n')


def main():
    paths = [path for _, path in METHODS_PATH]
    paths.extend([f"{path}\\partial_func" for _, path in METHODS_PATH])
    add_floders(tuple(paths))



    for func in display_func.funcs:
        func()

    # compare a conditioned matrix and an unconditioned matrix for a quadratic function
    # comparaison_condi(get_other_diago(1000), np.array([[0] for _ in range(1000)]), 1e-10, int(2e3))


if __name__ == '__main__':
    main()
