from matplotlib import pyplot as plt
import numpy as np
from function import get_other_diago, get_zvankin_quad, condi_A
from display import (display_convergence_2d,
                     display_partial_func,
                     display_norm)
from methods import (gradient_descent_fix_step,
                     gradient_descent_optimal_step,
                     quadratic_gradient_descent,
                     quadratic_conjuguate_gradient_method,
                     comparaison_condi)
from tools import add_floders



def main():
    add_floders()

    display_convergence_2d('figure\\grad_desc_fix_step',
                            get_zvankin_quad(2),
                            np.array([[-5], [-5]]),
                            gradient_descent_fix_step)

    display_convergence_2d('figure\\grad_desc_optimal_step',
                            get_zvankin_quad(2),
                            np.array([[-5], [-5]]),
                            gradient_descent_optimal_step)

    display_partial_func('figure\\grad_desc_fix_step\\partial_func',
                          get_zvankin_quad(2),
                          np.array([[-5], [-5]]),
                          gradient_descent_fix_step)
    
    display_partial_func('figure\\grad_desc_optimal_step\\partial_func',
                         get_zvankin_quad(2),
                         np.array([[-5], [-5]]),
                         gradient_descent_optimal_step)
    comparaison_condi(get_other_diago(1000), np.array([[0] for i in range(1000)]), 1e-10, 2*10**3)



if __name__ == '__main__':
    main()
