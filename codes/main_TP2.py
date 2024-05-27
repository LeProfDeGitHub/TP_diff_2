from matplotlib import pyplot as plt
from matplotlib import colors as plt_color
import numpy as np
from function import get_J_1, get_J_2
from opti_methods import (gradient_descent_optimal_step)
from display_TP1 import (display_convergence_2d, plot_contour)

J1 = get_J_1(np.array([[1.0, -1.0],
                       [3.0,  0.0]]),
             np.array([[-0.1],
                       [-0.1]]))

plot_contour(J1, (-5, 10), (-10, 5), k=np.linspace(0, 10, 100))
plt.grid()
plt.show()
