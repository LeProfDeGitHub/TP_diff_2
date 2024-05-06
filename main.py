import numpy as np
from function import get_zvankin_quad
from display import display_convergence_2d
from methods import gradient_descent_fix_step, gradient_descent_optimal_step



def main():
    display_convergence_2d('figure\\grad_desc_fix_step',
                            get_zvankin_quad(2),
                            np.array([[0], [0]]),
                            gradient_descent_fix_step)

    display_convergence_2d('figure\\grad_desc_optimal_step',
                            get_zvankin_quad(2),
                            np.array([[0], [0]]),
                            gradient_descent_optimal_step)



if __name__ == '__main__':
    main()
