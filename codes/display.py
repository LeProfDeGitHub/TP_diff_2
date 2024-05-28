import numpy as np

from matplotlib import pyplot as plt
from matplotlib import contour as ctr

from function import Function, get_other_diago, condi_A
from opti_methods import METHODE_TYPE


def plot_contour(f: Function, x_space: np.ndarray, y_space: np.ndarray, z_space: np.ndarray | None = None, norm = None):
    x, y = np.meshgrid(x_space, y_space)

    z = np.array([f(np.array([[x[i, j]], [y[i, j]]])) for i in range(100) for j in range(100)]).reshape(100, 100)

    if z_space is None:
        z_space = np.linspace(np.nanmin(z), np.nanmax(z), (len(x_space) + len(y_space)) // 2)

    plt.clf()
    contour = plt.contourf(x, y, z, levels=z_space, cmap='viridis', alpha=1, norm=norm)
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.colorbar(contour, label='f(x, y)')


def display_convergence_2d(path: str, J: Function, methode: METHODE_TYPE, X0: np.ndarray,
                           x_space: np.ndarray, y_space: np.ndarray, z_space: np.ndarray | None = None,
                           eps: float = 5e-2, niter: int = 1000):
    Xn = methode(J, X0, eps, niter)
    
    plot_contour(J, x_space, y_space, z_space)
    plt.plot(Xn[:, 0], Xn[:, 1], 'r*--', label='Gradient Descente')
    plt.savefig(f'{path}\\convergence_grad_de.png')
    print(f'File saved at {path}\\convergence_grad_de.png')


def display_convergence_by_X0(path: str, J: Function, methode: METHODE_TYPE,
                              x_space: np.ndarray, y_space: np.ndarray,
                              eps: float = 5e-2, niter: int = 1000):
    x, y = np.meshgrid(x_space, y_space)
    ngrid = (len(x_space) + len(y_space)) // 2
    Z = np.array([[len(methode(J, np.array([[x[i, j]], [y[i, j]]]), eps, niter)) - 1 for i in range(ngrid)] for j in range(ngrid)])

    plt.clf()
    plt.imshow(Z, extent=[x_space[0], x_space[-1], y_space[0], y_space[-1]], origin='lower') # type: ignore
    plt.colorbar(label='Nombre d\'itérations', ticks=np.linspace(np.nanmin(Z), np.nanmax(Z), 10, dtype=int))
    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig(f'{path}\\convergence_by_X0.png')
    print(f'File saved at {path}\\convergence_by_X0.png')


def display_partial_func(path: str, J: Function, methode: METHODE_TYPE, X0: np.ndarray):
    Xn = methode(J, X0, 5e-2, 1000)
    
    i_lnspace = np.linspace(1, len(Xn), 10, dtype=int, endpoint=False)
    Xn        = Xn[i_lnspace]
    grad      = [J.df(X) for X in Xn]
    grad_norm = [grad_val / np.linalg.norm(grad_val) for grad_val in grad]

    xn  = np.linspace(-10, 10, 1000)
    yns = np.array([[J.partial(X, d)(x)[0, 0] for x in xn]
                    for X, d in zip(Xn, grad_norm)])

    segs = [[[[x, y] for x, y in zip(xn, yn)]] for yn in yns]

    cmap = plt.get_cmap('Blues')
    
    plt.clf()
    plt.title(f'Coupes de la fonction')
    cs = ctr.ContourSet(plt.gca(), i_lnspace, segs, cmap=cmap)
    # plt.clabel(cs, inline=True, fontsize=8, fmt=lambda x: f'i = {x:.2f}')
        # plt.contour(xn, xn, np.array([[J(np.array([[x], [y]])) for x in xn] for y in xn]), cmap=cmap, alpha=0.5)

    plt.savefig(f'{path}\\partial_functs.png')
    print(f'File saved at {path}\\partial_functs.png')


    for iy, i in enumerate(i_lnspace):
        plt.clf()
        plt.title(f'Coupe de la fonction ({i+1}/{len(Xn) + 1})')
        plt.plot(xn, yns[iy])
        plt.xlabel('t')
        plt.ylabel(r'$f(x_i - t\nabla f(x_i))$')
        plt.savefig(f'{path}\\partial_funct({i+1}).png')


def display_norm(path: str, J: Function, methode: METHODE_TYPE, X0: np.ndarray):
    Xn = methode(J, X0, 5e-2, 1000)
    
    grad = [J.df(X) for X in Xn]

    plt.clf()
    plt.title('Convergence de la solution')
    plt.plot(np.arange(len(Xn)), [np.linalg.norm(grad_val) for grad_val in grad])
    plt.xlabel('Nombre d\'itérations $i$')
    plt.ylabel(r'$\|\nabla f(x_i)\|$')
    plt.xticks(np.linspace(0, len(Xn)-1, 10, dtype=int))
    plt.grid()
    plt.savefig(f'{path}\\convergence.png')
    print(f'File saved at {path}\\convergence.png')


def display_error(path: str, J: Function, methode: METHODE_TYPE, X0: np.ndarray, solution: np.ndarray):
    Xn = methode(J, X0, 5e-2, 20_000)
    
    err = np.array([np.linalg.norm(x - solution) for x in Xn])

    plt.clf()
    plt.semilogy(np.arange(len(Xn)), err)
    plt.xlabel('Nombre d\'itérations $i$')
    plt.ylabel(r'$\|x_i-x^*\|$')
    plt.xticks(np.linspace(0, len(Xn)-1, 10, dtype=int))
    plt.grid()
    plt.savefig(f'{path}\\error.png')
    print(f'File saved at {path}\\error.png')


def display_compare_error(path: str, J: Function,
                          methodes_labels: tuple[tuple[METHODE_TYPE, str], ...],
                          X0: np.ndarray, solution: np.ndarray):
    plt.clf()
    for methode, label in methodes_labels:
        Xn = methode(J, X0, 5e-2, 20_000)
    
        err = np.array([np.linalg.norm(x - solution) for x in Xn])

        plt.loglog(np.arange(len(Xn)), err, label=label)
    plt.legend()
    plt.grid()
    plt.xlabel('Nombre d\'itérations $i$')
    plt.ylabel(r'\|x_i-x^*\|')
    plt.grid()
    plt.savefig(f'{path}\\error.png')
    print(f'File saved at {path}\\error.png')


def display_ka(path: str, nmax: int ) :
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
    plt.grid()
    plt.savefig(path)
    print(f'File saved at {path}')
