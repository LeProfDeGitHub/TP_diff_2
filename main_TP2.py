import autograd.numpy as np
from function import get_J_1, get_J_2
from methods import (gradient_descent_optimal_step)
from display import (display_convergence_2d)

J1 = get_J_1(np.array([[ 1.0, 3.0],
                       [-1.0, 0.0]]),
             np.array([[-0.1],
                       [-0.1]]))

J2 = get_J_2(2)

display_convergence_2d('figure', J1, np.array([[-5.0], [5.0]]), gradient_descent_optimal_step)




