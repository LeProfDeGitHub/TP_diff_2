from matplotlib import pyplot as plt
from matplotlib import colors as plt_color
import numpy as np
from tools import (TestFuncsCollection, add_floders, init_figure_folder, log_range,
                   test_deco_n)
from function import (get_J_1,
                      get_J_2,
                      gen_J_1,)
from opti_methods import (METHODS_LABEL_PATH, gradient_descent_fix_step,
                          gradient_descent_optimal_step,
                          newton,
                          newton_optimal_step,
                          BFGS,
                          DFP,)
from display import (display_convergence_by_X0,
                     display_norm,
                     display_compare_norm,
                     display_time_N,)


METHODS = (
    gradient_descent_fix_step,
    gradient_descent_optimal_step,
    newton,
    newton_optimal_step,
    BFGS,
    DFP,
)

# Objects to store the functions to be executed.
func_collection = TestFuncsCollection("TP2")

nbr_methods = len(METHODS)

X0 = np.array([[-5], [1]])

# A = np.random.randn(2, 2)
# b = np.array([[-0.1], [-0.1]])
# J = get_J_1(A, b)

# Xn = newton_optimal_step(J, X0, eps=1e-6, niter=1000)
# z = J(Xn[-1])


@test_deco_n(func_collection, nbr_methods)
def display_all_convergence_by_X0(test_funcs_collection: TestFuncsCollection):
    '''
    Call display_convergence_by_X0 for each method in METHODS_PATH.
    '''
    A = np.random.randn(2, 2)
    b = np.array([[-0.1], [-0.1]])
    J = get_J_1(A, b)
    for method in METHODS:
        _, path = METHODS_LABEL_PATH[method]
        test_funcs_collection.current_nbr += 1
        test_funcs_collection.print_current_nbr()
        display_convergence_by_X0(f'figure\\{path}',
                                  J, method,
                                  np.linspace(-10, 10, 10),
                                  np.linspace(-10, 10, 10),
                                  eps = 0.1)

@test_deco_n(func_collection, nbr_methods)
def display_all_norm(test_funcs_collection: TestFuncsCollection):
    '''
    Call display_norm for each method in METHODS_PATH.
    '''
    A = np.random.randn(2, 2)
    b = np.array([[-0.1], [-0.1]])
    J = get_J_1(A, b)
    for method in METHODS:
        _, path = METHODS_LABEL_PATH[method]
        test_funcs_collection.current_nbr += 1
        test_funcs_collection.print_current_nbr()
        display_norm(f'figure\\{path}',
                      J, method,
                      X0)

@test_deco_n(func_collection, 1)
def display_all_compare_norm(test_funcs_collection: TestFuncsCollection):
    '''
    Call display_compare_norm for each method in METHODS_PATH.
    '''

    methods_label = tuple((method, METHODS_LABEL_PATH[method][0]) for method in METHODS)

    A = np.random.randn(2, 2)
    b = np.array([[-0.1], [-0.1]])
    J = get_J_1(A, b)

    X0 = np.array([[-5], [1]])

    test_funcs_collection.current_nbr += 1
    test_funcs_collection.print_current_nbr()
    display_compare_norm(f'figure',
                         J, methods_label,
                         X0,)

@test_deco_n(func_collection, nbr_methods)
def display_all_time_N(test_funcs_collection: TestFuncsCollection):
    '''
    Call display_time_N for each method in METHODS_PATH.
    '''
    n_space = log_range(1, 100, 10)
    n_space = np.array([int(n) for n in n_space])
    for method in METHODS:
        _, path = METHODS_LABEL_PATH[method]
        test_funcs_collection.current_nbr += 1
        test_funcs_collection.print_current_nbr()
        display_time_N(f'figure\\{path}', gen_J_1, method, n_space)



def main():
    init_figure_folder()
    paths = [METHODS_LABEL_PATH[method][1] for method in METHODS]
    add_floders(tuple(paths))

    for func in func_collection.funcs:
        func()

    

if __name__ == '__main__':
    main()
