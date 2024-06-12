from matplotlib import pyplot as plt
from matplotlib import colors as plt_color
import numpy as np
import scipy
from tools import (TestFuncsCollection,
                   add_floders,
                   init_figure_folder,
                   log_range,
                   test_deco_n,
                   time_func,)
from function import (get_J_1,
                      get_J_2,
                      gen_J_1,)
from opti_methods import (METHODS_LABEL_PATH, gradient_descent_fix_step,
                          gradient_descent_optimal_step,
                          newton,
                          newton_optimal_step,
                          quasi_newton,
                          quasi_newton_BFGS,
                          quasi_newton_DFP)
from display import (display_convergence_by_X0,
                     display_norm,
                     display_compare_norm,
                     display_time_N,
                     display_error,
                     display_error_J,
                     display_error_J_solution)


METHODS = (
    quasi_newton_BFGS,
    quasi_newton_DFP,
    gradient_descent_fix_step,
    gradient_descent_optimal_step,
    newton,
    newton_optimal_step,
)

# Objects to store the functions to be executed.
func_collection = TestFuncsCollection("TP2")

nbr_methods = len(METHODS)

X0 = np.array([[-5], [1]])


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

@test_deco_n(func_collection, nbr_methods)
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
    # n_space = log_range(1, 100, 10)
    n_space = np.linspace(1, 10, 10, dtype=int)
    for method in METHODS:
        _, path = METHODS_LABEL_PATH[method]
        test_funcs_collection.current_nbr += 1
        test_funcs_collection.print_current_nbr()
        display_time_N(f'figure\\{path}', gen_J_1, method, n_space)

@test_deco_n(func_collection, nbr_methods)
def display_all_error(test_funcs_collection: TestFuncsCollection):
    '''
    Call display_error for each method in METHODS_PATH.
    '''
    n = 2
    J = gen_J_1(n)
    X0 = np.array([[1] for _ in range(n)])
    x_solu = scipy.optimize.minimize(J.f, np.random.randn(n), jac=J.df, method='L-BFGS-B', options={'disp': True}).x
    for method in METHODS:
        _, path = METHODS_LABEL_PATH[method]
        test_funcs_collection.current_nbr += 1
        test_funcs_collection.print_current_nbr()
        display_error(f'figure\\{path}', J, method, X0, x_solu)

@test_deco_n(func_collection, nbr_methods)
def display_all_error_J(test_funcs_collection: TestFuncsCollection):
    '''
    Call display_error_J for each method in METHODS_PATH.
    '''
    n_space = log_range(1, 100, 10)
    n_space = np.array([int(n) for n in n_space])
    J = gen_J_1(2)
    X0 = np.array([[1] for _ in range(2)])
    solu = scipy.optimize.minimize(J.f, X0.reshape(-1), jac=J.df, method='L-BFGS-B').x
    for method in METHODS:
        _, path = METHODS_LABEL_PATH[method]
        test_funcs_collection.current_nbr += 1
        test_funcs_collection.print_current_nbr()
        display_error_J(f'figure\\{path}', J, method, X0, solu)





def main():
    # init_figure_folder()
    paths = [METHODS_LABEL_PATH[method][1] for method in METHODS]
    # add_floders(tuple(paths))

    run = time_func(func_collection.run)
    delta_time, _ = run()

    print(f"Total time: {delta_time:.2f} s")

    

if __name__ == '__main__':
    main()
    # n = 2
    # J = gen_J_1(n)
    # x_solu = scipy.optimize.minimize(J.f, np.random.randn(n), jac=J.df, method='L-BFGS-B', options={'disp': True}).x
    # print(x_solu)
