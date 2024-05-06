from matplotlib import pyplot as plt
import numpy as np
from function import get_zvankin_quad , tri_diagonal
from display import plot_contour
from methods import gradient_descent_optimal_step








plt.cla()
plt.title('Convergence de la solution')
plt.plot(np.arange(i_max+1), [np.linalg.norm(grad_val) for grad_val in grad])
plt.show()


