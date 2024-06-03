import numpy as np
from function import QuadraticFunction, get_other_diago, get_zvankin_quad, condi_A
from display import (display_convergence_2d,
                     display_convergence_by_X0,
                     display_partial_func,
                     display_norm,
                     display_error,
                     display_compare_error,
                     display_ka)
from opti_methods import (METHOD_TYPE,
                          METHODS_LABEL_PATH,
                          gradient_descent_fix_step, newton, newton_optimal_step, gradient_descent_optimal_step,
                          quadratic_gradient_descent_optimal_step,
                          quadratic_conjuguate_gradient_method,)
from tools import (init_figure_folder,
                   add_floders,
                   TestFuncsCollection,
                   test_deco_n,)



def comparaison_condi(f: QuadraticFunction, X0, eps: float, niter: int):
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
    X_gd = quadratic_gradient_descent_optimal_step(f, X0, eps, niter)
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
    X_gd = quadratic_gradient_descent_optimal_step(quad, X0, eps, niter)
    print(f'gradient descent method optimal step: {len(X_gd)} iterations')
    error = np.linalg.norm(quad.df (X_gd[-1]))
    print(f'error: {error }\n')

    X_cg = quadratic_conjuguate_gradient_method(quad, X0, eps, niter)
    print(f'conjuguate gradient method: {len(X_cg)} iterations')
    error = np.linalg.norm(quad.df (X_cg[-1]))
    print(f'error: {error}\n')




METHODS: tuple[METHOD_TYPE, ...] = (
    gradient_descent_fix_step,
    quadratic_gradient_descent_optimal_step,
    quadratic_conjuguate_gradient_method,
    newton,
    newton_optimal_step,
    gradient_descent_optimal_step,
)

# Objects to store the functions to be executed.
func_collection = TestFuncsCollection("TP1")

nbr_methods = len(METHODS)

X0 = np.array([[-5], [1]])

@test_deco_n(func_collection, nbr_methods)
def display_all_converence_2d(test_funcs_collection: TestFuncsCollection):
    '''
    Call display_convergence_2d for each method in METHODS_PATH.
    '''
    J = get_zvankin_quad(2)
    for method in METHODS:
        _, path = METHODS_LABEL_PATH[method]
        test_funcs_collection.current_nbr += 1
        test_funcs_collection.print_current_nbr()
        display_convergence_2d(f'figure\\{path}',
                               J, method, X0,
                               np.linspace(-10, 10, 100),
                               np.linspace(-10, 10, 100))

@test_deco_n(func_collection, nbr_methods)
def display_all_converence_by_X0(test_funcs_collection: TestFuncsCollection):
    '''
    Call display_convergence_by_X0 for each method in METHODS_PATH.
    '''
    J = get_zvankin_quad(2)
    for method in METHODS:
        _, path = METHODS_LABEL_PATH[method]
        test_funcs_collection.current_nbr += 1
        test_funcs_collection.print_current_nbr()
        display_convergence_by_X0(f'figure\\{path}',
                                  J, method,
                                  np.linspace(-10, 10, 50),
                                  np.linspace(-10, 10, 50))

@test_deco_n(func_collection, nbr_methods)
def display_all_partial_func(test_funcs_collection: TestFuncsCollection):
    '''
    Call display_partial_func for each method in METHODS_PATH.
    '''
    J = get_zvankin_quad(2)
    for method in METHODS:
        _, path = METHODS_LABEL_PATH[method]
        test_funcs_collection.current_nbr += 1
        test_funcs_collection.print_current_nbr()
        display_partial_func(f'figure\\{path}\\partial_func',
                             J, method, X0)

@test_deco_n(func_collection, nbr_methods)
def display_all_norm(test_funcs_collection: TestFuncsCollection):
    '''
    Call display_norm for each method in METHODS_PATH.
    '''
    J = get_zvankin_quad(2)
    for method in METHODS:
        _, path = METHODS_LABEL_PATH[method]
        test_funcs_collection.current_nbr += 1
        test_funcs_collection.print_current_nbr()
        display_norm(f'figure\\{path}',
                     J, method, X0)

@test_deco_n(func_collection, nbr_methods)
def display_all_error(test_funcs_collection: TestFuncsCollection):
    '''
    Call display_error for each method in METHODS_PATH.
    '''
    J = get_zvankin_quad(2)
    x_solu = np.linalg.solve(J.A, J.b)
    for method in METHODS:
        _, path = METHODS_LABEL_PATH[method]
        test_funcs_collection.current_nbr += 1
        test_funcs_collection.print_current_nbr()
        display_error(f'figure\\{path}',
                      J, method, X0,
                      x_solu)

@test_deco_n(func_collection, 1)
def display_all_compare_error(test_funcs_collection: TestFuncsCollection):
    '''
    Call display_compare_error for each method in METHODS_LABELS.
    '''

    methods_label = tuple((method, METHODS_LABEL_PATH[method][0]) for method in METHODS)

    J = get_zvankin_quad(2)
    x_solu = np.linalg.solve(J.A, J.b)
    test_funcs_collection.current_nbr += 1
    test_funcs_collection.print_current_nbr()
    display_compare_error(f'figure\\',
                          J, methods_label, X0,
                          x_solu)

@test_deco_n(func_collection, 1)
def display_ka_wrp(test_funcs_collection: TestFuncsCollection):
    '''
    Call display_ka for the condition number of the matrix A.
    '''
    test_funcs_collection.current_nbr += 1
    test_funcs_collection.print_current_nbr()
    display_ka('figure\\condition_number.png', 500)

@test_deco_n(func_collection, 1)
def comparaison_condi_wrp(test_funcs_collection: TestFuncsCollection):
    """
    Call comparaison_condi for a conditioned matrix and an unconditioned matrix for a quadratic function.
    """
    test_funcs_collection.current_nbr += 1
    test_funcs_collection.print_current_nbr()
    print()
    comparaison_condi(get_other_diago(1000), np.array([[0] for _ in range(1000)]), 1e-10, int(2e3))


def main():
    init_figure_folder()
    paths = [METHODS_LABEL_PATH[method][1] for method in METHODS]
    paths.extend([f"{path}\\partial_func" for path in paths])
    add_floders(tuple(paths))

    for func in func_collection.funcs:
        func()


if __name__ == '__main__':
    main()
