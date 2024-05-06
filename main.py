from matplotlib import pyplot as plt
import numpy as np
from function import get_zvankin_quad , tri_diagonal
from display import plot_contour
from methods import quadratic_gradient_descent, quadratic_conjuguate_gradient_method








plt.cla()
plt.title('Convergence de la solution')
plt.plot(np.arange(i_max+1), [np.linalg.norm(grad_val) for grad_val in grad])
plt.show()

# gradient descent method

J2 = tri_diagonal(1000)
#optimal step gradient descent
X_gd, i_max = quadratic_gradient_descent(J2, np.zeros((1000, 1)), 1e-10, 2*10**3)

print(f'gradient descent method optimal step: {i_max} iterations')

# conjuguate gradient method
Xgd, i_max = quadratic_conjuguate_gradient_method(J2, np.zeros((1000, 1)), 1e-10, 2*10**3)

print(f'conjuguate gradient method: {i_max} iterations')



