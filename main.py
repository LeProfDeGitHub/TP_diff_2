from matplotlib import pyplot as plt
import numpy as np
from function import get_zvankin_quad , tri_diagonal
from display import (display_convergence_2d,
                     display_partial_func,
                     display_norm)
from methods import (gradient_descent_fix_step,
                     gradient_descent_optimal_step, quadratic_gradient_descent, quadratic_conjuguate_gradient_method)



def main():
    display_convergence_2d('figure\\grad_desc_fix_step',
                            get_zvankin_quad(2),
                            np.array([[0], [0]]),
                            gradient_descent_fix_step)

    display_convergence_2d('figure\\grad_desc_optimal_step',
                            get_zvankin_quad(2),
                            np.array([[0], [0]]),
                            gradient_descent_optimal_step)

    display_partial_func('figure\\grad_desc_fix_step\\partial_func',
                          get_zvankin_quad(2),
                          np.array([[0], [0]]),
                          gradient_descent_fix_step)
    
    display_partial_func('figure\\grad_desc_optimal_step\\partial_func',
                         get_zvankin_quad(2),
                         np.array([[0], [0]]),
                         gradient_descent_optimal_step)

    # gradient descent method

    J2 = tri_diagonal(1000)
    #optimal step gradient descent
    X_gd, i_max = quadratic_gradient_descent(J2, np.zeros((1000, 1)), 1e-10, 2*10**3)

    print(f'gradient descent method optimal step: {i_max} iterations')

    # conjuguate gradient method
    Xgd, i_max = quadratic_conjuguate_gradient_method(J2, np.zeros((1000, 1)), 1e-10, 2*10**3)

    print(f'conjuguate gradient method: {i_max} iterations')





if __name__ == '__main__':
    main()
