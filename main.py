from matplotlib import pyplot as plt
import numpy as np
from function import get_zvankin_quad
from display import plot_contour
from methods import gradient_descent

J = get_zvankin_quad(2)
X_gd, _ = gradient_descent(J, np.array([[0], [0]]), 1e-3, 1000)

plot_contour(J, (-10, 10), (-10, 10))

plt.plot(X_gd[:, 0], X_gd[:, 1], 'r*--', label='Gradient Descente')
plt.savefig('figure/gradient_descent.svg') #enregistre la figure
plt.show()



x         = np.linspace(-10, 10, 1000)
grad      = [J.df(X) for X in X_gd]
grad_norm = [grad_val / np.linalg.norm(grad_val) for grad_val in grad]

J0 = [J.partial(X_gd[0], grad_norm[0])(x_val) for x_val in x]

y = np.array([[J.partial(X, d)(x_val)[0, 0] for x_val in x]
              for X, d in zip(X_gd, grad_norm)])

for i, y_val in enumerate(y):
    plt.cla()
    plt.title(f'Coupe de la fonction ({i+1}/{len(y)})')
    plt.plot(x, y_val)
    plt.savefig(f'figure/Coupe de la fonction ({i+1}).svg')
    plt.show()
