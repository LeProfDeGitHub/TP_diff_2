from matplotlib import pyplot as plt
import autograd.numpy as np

from function import Function, QuadraticFunction, get_other_diago, condi_A
from matplotlib import pyplot as plt
from matplotlib import colors as plt_color

from methods import METHODE_TYPE



def plot_contour(f: Function, xlim: tuple[float, float], ylim: tuple[float, float], norm=None):
    x, y = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 100),
        np.linspace(ylim[0], ylim[1], 100)
    )
    z = np.array([f(np.array([[x[i, j]], [y[i, j]]])) for i in range(100) for j in range(100)]).reshape(100, 100)

    k = np.linspace(np.nanmin(z), np.nanmax(z), 100)

    plt.clf()
    contour = plt.contourf(x, y, z, levels=k, cmap='viridis', alpha=1, norm=norm)
    
    plt.colorbar(contour, label='f(x, y)')



def display_convergence_2d(path: str, J: QuadraticFunction, X0: np.ndarray, methode: METHODE_TYPE):
    Xn, _ = methode(J, X0, 1e-3, 1000)
    
    plot_contour(J, (-10, 10), (-10, 10))
    plt.plot(Xn[:, 0], Xn[:, 1], 'r*--', label='Gradient Descente')
    plt.savefig(f'{path}\\convergence_grad_de.png')



def display_partial_func(path: str, J: QuadraticFunction, X0: np.ndarray, methode: METHODE_TYPE):
    Xn, imax = methode(J, X0, 1e-3, 1000)
    
    x         = np.linspace(-10, 10, 1000)
    grad      = [J.df(X) for X in Xn]
    grad_norm = [grad_val / np.linalg.norm(grad_val) for grad_val in grad]

    y = np.array([[J.partial(X, d)(x_val)[0, 0] for x_val in x]
                for X, d in zip(Xn, grad_norm)])

    for i in np.linspace(0, imax, 10, dtype=int, endpoint=False):
        plt.clf()
        plt.title(f'Coupe de la fonction ({i+1}/{imax + 1})')
        plt.plot(x, y[i])
        plt.savefig(f'{path}\\partial_funct({i+1}).png')


def display_norm(path: str, J: QuadraticFunction, X0: np.ndarray, methode: METHODE_TYPE):

    Xn, imax = methode(J, X0, 1e-3, 1000)
    
    grad = [J.df(X) for X in Xn]

    plt.clf()
    plt.title('Convergence de la solution')
    plt.plot(np.arange(imax + 1), [np.linalg.norm(grad_val) for grad_val in grad])
    plt.savefig(f'{path}\\convergence.png')

def display_error(path: str, J: QuadraticFunction, X0: np.ndarray, methode: METHODE_TYPE):

    Xn, imax = methode(J, X0, 1e-5, 20_000)
    
    err = np.array([np.linalg.norm(J.A @ x - J.b) for x in Xn])

    plt.clf()
    plt.title('Erreur de la solution')
    plt.semilogy(np.arange(imax), err)
    plt.savefig(f'{path}\\error.png')

def display_compare_error(path: str, J: QuadraticFunction, X0: np.ndarray,
                          methodes_labels: list[tuple[METHODE_TYPE, str]]):
    plt.clf()
    for methode, label in methodes_labels:
        Xn, imax = methode(J, X0, 1e-5, 20_000)
    
        err = np.array([np.linalg.norm(J.A @ x - J.b) for x in Xn])

        plt.title('Erreur de la solution')
        plt.loglog(np.arange(imax), err, label=label)
    plt.savefig(f'{path}\\error.png')


def display_ka( nmax :int ) :
    """
    display the condition number of the matrix A
    :param f: QuadraticFunction
    :param nmax: max size of the matrix
    """
    ka = []
    for i in range(1, nmax):
        A = get_other_diago(i)
        f = condi_A(A)
        ka.append(np.linalg.cond(f.A))

    plt.clf()
    plt.title('Condition number of the matrix A')
    plt.plot(range(1, nmax), ka)
    plt.savefig('figure\\condition_number.png')

#test
display_ka(10)
