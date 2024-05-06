from matplotlib import pyplot as plt
import numpy as np

from function import Function, QuadraticFunction
from matplotlib import pyplot as plt
from matplotlib import colors as plt_color

from methods import METHODE_TYPE



def plot_contour(f: Function, xlim: tuple[float, float], ylim: tuple[float, float], norm=None):
    x, y = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 100),
        np.linspace(ylim[0], ylim[1], 100)
    )

    z = np.array([f(np.array([[x[i, j]], [y[i, j]]]))[0] for i in range(100) for j in range(100)]).reshape(100, 100)

    k = np.linspace(np.nanmin(z), np.nanmax(z), 100)

    plt.clf()
    contour = plt.contourf(x, y, z, levels=k, cmap='viridis', alpha=1, norm=norm)
    
    plt.colorbar(contour, label='f(x, y)')



def display_convergence_2d(path: str, J: QuadraticFunction, X0: np.ndarray, methode: METHODE_TYPE):
    X_gd, _ = methode(J, X0, 1e-3, 1000)
    
    plot_contour(J, (-10, 10), (-10, 10))
    plt.plot(X_gd[:, 0], X_gd[:, 1], 'r*--', label='Gradient Descente')
    plt.savefig(f'{path}\\convergence_grad_de.png')



def display_partial_func(path: str, J: QuadraticFunction, X0: np.ndarray, methode: METHODE_TYPE):
    X_gd, _ = methode(J, X0, 1e-3, 1000)
    
    x         = np.linspace(-10, 10, 1000)
    grad      = [J.df(X) for X in X_gd]
    grad_norm = [grad_val / np.linalg.norm(grad_val) for grad_val in grad]

    y = np.array([[J.partial(X, d)(x_val)[0, 0] for x_val in x]
                for X, d in zip(X_gd, grad_norm)])

    for i in np.linspace(0, len(y), 10, dtype=int, endpoint=False):
        plt.clf()
        plt.title(f'Coupe de la fonction ({i+1}/{len(y)})')
        plt.plot(x, y[i])
        plt.savefig(f'{path}\\partial_funct({i+1}).png')


def display_norm(path: str, J: QuadraticFunction, X0: np.ndarray, methode: METHODE_TYPE):

    X_gd, _ = methode(J, X0, 1e-3, 1000)
    
    grad = [J.df(X) for X in X_gd]

    plt.clf()
    plt.title('Convergence de la solution')
    plt.plot(np.arange(len(grad)), [np.linalg.norm(grad_val) for grad_val in grad])
    plt.savefig(f'{path}\\convergence.png')
