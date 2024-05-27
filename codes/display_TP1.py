import numpy as np

from matplotlib import pyplot as plt
from matplotlib import contour as ctr

from function import Function, QuadraticFunction, get_other_diago, condi_A
from opti_methods import QUAD_METHODE_TYPE


def plot_contour(f: Function, xlim: tuple[float, float], ylim: tuple[float, float], norm = None,
                 k: np.ndarray | None = None):
    x, y = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 100),
        np.linspace(ylim[0], ylim[1], 100)
    )
    z = np.array([f(np.array([[x[i, j]], [y[i, j]]])) for i in range(100) for j in range(100)]).reshape(100, 100)

    if k is None:
        k = np.linspace(np.nanmin(z), np.nanmax(z), 100)

    plt.clf()
    contour = plt.contourf(x, y, z, levels=k, cmap='viridis', alpha=1, norm=norm)
    
    plt.colorbar(contour, label='f(x, y)')


def display_convergence_2d(path: str, J: QuadraticFunction, X0: np.ndarray, methode: QUAD_METHODE_TYPE):
    Xn = methode(J, X0, 5e-2, 1000)
    
    plot_contour(J, (-10, 10), (-10, 10))
    plt.plot(Xn[:, 0], Xn[:, 1], 'r*--', label='Gradient Descente')
    plt.savefig(f'{path}\\convergence_grad_de.png')
    print(f'File saved at {path}\\convergence_grad_de.png')


def display_convergence_by_X0(path: str, J: QuadraticFunction, xlim: tuple[float, float], ylim: tuple[float, float],
                              ngrid: int, eps: float, niter: int, methode: QUAD_METHODE_TYPE):
    x, y = np.meshgrid(
        np.linspace(xlim[0], xlim[1], ngrid),
        np.linspace(ylim[0], ylim[1], ngrid)
    )
    Z = np.array([[len(methode(J, np.array([[x[i, j]], [y[i, j]]]), eps, niter)) for i in range(ngrid)] for j in range(ngrid)])

    plt.clf()
    plt.imshow(Z, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower') # type: ignore
    plt.colorbar(label='Nombre d\'itérations')
    plt.savefig(f'{path}\\convergence_by_X0.png')
    print(f'File saved at {path}\\convergence_by_X0.png')


def display_partial_func(path: str, J: QuadraticFunction, X0: np.ndarray, methode: QUAD_METHODE_TYPE):
    Xn = methode(J, X0, 5e-2, 1000)
    
    
    i_lnspace = np.linspace(0, len(Xn), 10, dtype=int, endpoint=False)
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
        plt.savefig(f'{path}\\partial_funct({i+1}).png')


def display_norm(path: str, J: QuadraticFunction, X0: np.ndarray, methode: QUAD_METHODE_TYPE):

    Xn = methode(J, X0, 5e-2, 1000)
    
    grad = [J.df(X) for X in Xn]

    plt.clf()
    plt.title('Convergence de la solution')
    plt.plot(np.arange(len(Xn)), [np.linalg.norm(grad_val) for grad_val in grad])
    plt.savefig(f'{path}\\convergence.png')
    print(f'File saved at {path}\\convergence.png')


def display_error(path: str, J: QuadraticFunction, X0: np.ndarray, methode: QUAD_METHODE_TYPE):

    Xn = methode(J, X0, 5e-2, 20_000)
    
    err = np.array([np.linalg.norm(J.A @ x - J.b) for x in Xn])

    plt.clf()
    plt.title('Erreur de la solution')
    plt.semilogy(np.arange(len(Xn)), err)
    plt.savefig(f'{path}\\error.png')
    print(f'File saved at {path}\\error.png')


def display_compare_error(path: str, J: QuadraticFunction, X0: np.ndarray,
                          methodes_labels: tuple[tuple[QUAD_METHODE_TYPE, str], ...]):
    plt.clf()
    for methode, label in methodes_labels:
        Xn = methode(J, X0, 5e-2, 20_000)
    
        err = np.array([np.linalg.norm(J.A @ x - J.b) for x in Xn])

        plt.title('Erreur de la solution')
        plt.loglog(np.arange(len(Xn)), err, label=label)
    plt.legend()
    plt.grid()
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel('Erreur')
    plt.savefig(f'{path}\\error.png')
    print(f'File saved at {path}\\error.png')


def display_ka(path : str,  nmax :int ) :
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
