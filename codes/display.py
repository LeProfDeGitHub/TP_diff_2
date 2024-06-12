from typing import Callable, Literal
import numpy as np

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import contour as ctr
from matplotlib.colors import Normalize, LogNorm

from tools import time_func
from function import (Function,
                      get_other_diago,
                      condi_A,
                      phi)
from opti_methods import METHOD_TYPE


def plot_contour(f: Function, x_space: np.ndarray, y_space: np.ndarray, z_space: np.ndarray | None = None, norm = None):
    '''
    Plot the colored surface of a function with the following parameters:
    - `f: Function` a function object that has the `f` method
    - `x_space: np.ndarray` the x values of the grid
    - `y_space: np.ndarray` the y values of the grid
    - `z_space: np.ndarray | None` the z values allowed for the function
    - `norm: plt_color.Normalize | None` the normalization of the colors
    '''
    x, y = np.meshgrid(x_space, y_space)

    z = np.array([f(np.array([[x[i, j]], [y[i, j]]])) for i in range(100) for j in range(100)]).reshape(100, 100)

    if z_space is None:
        z_space = np.linspace(np.nanmin(z), np.nanmax(z), (len(x_space) + len(y_space)) // 2)

    plt.clf()
    contour = plt.contourf(x, y, z, levels=z_space, cmap='viridis', alpha=1, norm=norm)
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.colorbar(contour, label='f(x, y)')


def display_convergence_2d(path: str, J: Function, methode: METHOD_TYPE, X0: np.ndarray,
                           x_space: np.ndarray, y_space: np.ndarray, z_space: np.ndarray | None = None,
                           eps: float = 5e-2, niter: int = 1000):
    '''
    Display the colored surface of a function and the path of the gradient descent method with the following parameters:
    - `path: str` the path to save the figure
    - `J: Function` a function object that has the `f` method
    - `methode: METHODE_TYPE` the method to use
    - `X0: np.ndarray` the starting point of the method
    - `x_space: np.ndarray` the x values of the grid
    - `y_space: np.ndarray` the y values of the grid
    - `z_space: np.ndarray | None` the z values allowed for the function
    - `eps: float` the maximum error allowed
    - `niter: int` the maximum number of iterations allowed

    The function saves the figure at the path `path` and print a message to confirm the saving.
    '''
    Xn = methode(J, X0, eps, niter)
    grad = [J.df(X) for X in Xn]
    norm_grad = [grad_val/np.linalg.norm(grad_val) for grad_val in grad]
    
    plot_contour(J, x_space, y_space, z_space)
    plt.plot(Xn[:, 0], Xn[:, 1], 'r*--', label='Gradient Descente')

    for X, grad_val in zip(Xn, norm_grad):
        plt.arrow(X[0, 0], X[1, 0], grad_val[0, 0], grad_val[1, 0], head_width=0.5, head_length=0.5, fc='k', ec='k')

    plt.savefig(f'{path}\\convergence_grad_de.png')
    print(f'File saved at {path}\\convergence_grad_de.png')


def display_convergence_by_X0(path: str, J: Function, methode: METHOD_TYPE,
                              x_space: np.ndarray, y_space: np.ndarray,
                              eps: float = 5e-2, niter: int = 1000):
    '''
    Display on a colored surface the number of iterations needed to converge to the minimum
    for each starting point in the grid with the following parameters:
    - `path: str` the path to save the figure
    - `J: Function` a function object that has the `f` method
    - `methode: METHODE_TYPE` the method to use
    - `x_space: np.ndarray` the x values of the grid
    - `y_space: np.ndarray` the y values of the grid
    - `eps: float` the maximum error allowed
    - `niter: int` the maximum number of iterations allowed

    The function saves the figure at the path `path` and print a message to confirm the saving.
    '''
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


def display_partial_func(path: str, J: Function, methode: METHOD_TYPE, X0: np.ndarray):
    '''
    Display 10 partial functions of a function that are the cuts of the function along its gradient
    at 10 points of the gradient descent. Display also a single figure with all the cuts.
    The function takes the following parameters:
    - `path: str` the path to save the figures
    - `J: Function` a function object that has the `f` and `df` methods
    - `methode: METHODE_TYPE` the method to use
    - `X0: np.ndarray` the starting point of the method

    The function saves the figures at the path `path` and print a message to confirm the saving.
    '''
    Xn = methode(J, X0, 5e-2, 1000)
    nbr_iter = len(Xn)
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
    cs = ctr.ContourSet(plt.gca(), i_lnspace, segs, cmap=cmap)
    norm = Normalize(vmin=1, vmax=nbr_iter)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), label='i')
    # plt.clabel(cs, inline=True, fontsize=8, fmt=lambda x: f'i = {x:.2f}')
        # plt.contour(xn, xn, np.array([[J(np.array([[x], [y]])) for x in xn] for y in xn]), cmap=cmap, alpha=0.5)

    plt.savefig(f'{path}\\partial_functs.png')
    print(f'File saved at {path}\\partial_functs.png')


    for iy, i in enumerate(i_lnspace):
        plt.clf()
        plt.title(f'Coupe de la fonction ({i}/{nbr_iter})')
        plt.plot(xn, yns[iy])
        plt.xlabel('t')
        plt.ylabel(r'$f(x_i - t\nabla f(x_i))$')
        plt.savefig(f'{path}\\partial_funct({i}).png')


def display_norm(path: str, J: Function, methode: METHOD_TYPE, X0: np.ndarray,
                 log_mod: Literal['semilogx', 'semilogy', 'loglog'] | None = None):
    '''
    Display the norm of the gradient of the function at each iteration of a gradient descent method with the following parameters:
    - `path: str` the path to save the figure
    - `J: Function` a function object that has the `df` method
    - `methode: METHODE_TYPE` the method to use
    - `X0: np.ndarray` the starting point of the method
    - `log_mod: Literal['semilogx', 'semilogy', 'loglog'] | None` the log scale of the plot

    The function saves the figure at the path `path` and print a message to confirm the saving.
    '''
    Xn = methode(J, X0, 5e-2, 1000)
    
    grad = [J.df(X) for X in Xn]

    plt.clf()
    if log_mod is None:
        plt.plot(np.arange(len(Xn)), [np.linalg.norm(grad_val) for grad_val in grad])
    elif log_mod == 'semilogx':
        plt.semilogx(np.arange(len(Xn)), [np.linalg.norm(grad_val) for grad_val in grad])
    elif log_mod == 'semilogy':
        plt.semilogy(np.arange(len(Xn)), [np.linalg.norm(grad_val) for grad_val in grad])
    elif log_mod == 'loglog':
        plt.loglog(np.arange(len(Xn)), [np.linalg.norm(grad_val) for grad_val in grad])
    plt.xlabel('Nombre d\'itérations $i$')
    plt.ylabel(r'$\|\nabla f(x_i)\|$')
    plt.xticks(np.linspace(0, len(Xn)-1, 10, dtype=int))
    plt.grid()
    plt.savefig(f'{path}\\convergence.png')
    print(f'File saved at {path}\\convergence.png')


def display_error(path: str, J: Function, methode: METHOD_TYPE, X0: np.ndarray, solution: np.ndarray):
    '''
    Display the error of the gradient descent method at each iteration with the following parameters:
    - `path: str` the path to save the figure
    - `J: Function` a function object that has the `df` method
    - `methode: METHODE_TYPE` the method to use
    - `X0: np.ndarray` the starting point of the method
    - `solution: np.ndarray` the minimum of the function

    The function saves the figure at the path `path` and print a message to confirm the saving.
    '''
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


def display_error_N(path: str, J_gen: Callable[[int], Function], methode: METHOD_TYPE, n_space: np.ndarray):
    '''
    Display the error of the gradient descent method at each iteration for functions of different sizes with the following parameters:
    - `path: str` the path to save the figure
    - `J_gen: Callable[[int], Function]` a function that generate a function of size n
    - `methode: METHODE_TYPE` the method to use
    - `n_space: np.ndarray` the values of n

    The function saves the figure at the path `path` and print a message to confirm the saving.
    '''
    plt.clf()

    N_iter = np.zeros((len(n_space), 1))

    for i, N in enumerate(n_space):
        J = J_gen(N)
        X0 = np.array([[5] for _ in range(N)])
        Xn = methode(J, X0, 5e-2, 20_000)
        err = np.array([np.linalg.norm(x) for x in Xn])

        _, m = N_iter.shape
        new_N_iter = np.zeros((len(n_space), max(m, len(Xn))))
        new_N_iter[:, :] = 10e-10
        new_N_iter[:, :m] = N_iter
        N_iter = new_N_iter

        N_iter[i, :len(err)] = err[::-1]

    N_iter = N_iter[::-1, :]

    err_max = np.nanmax(N_iter)
    err_min = np.nanmin(N_iter)

    norm = LogNorm(vmin=10e-10, vmax=float(err_max))

    plt.imshow(N_iter, aspect='auto', norm=norm)
    plt.colorbar(label=r'$\log{\|x_i\|}$')
    plt.xlabel('Nombre d\'itérations $i$')
    plt.ylabel('Taille de la fonction $N$')
    plt.xticks(np.linspace(0, N_iter.shape[1]-1, 10, dtype=int), rotation=45)
    plt.yticks(np.arange(len(n_space)), [str(n) for n in n_space[::-1]])
    # plt.grid()
    plt.savefig(f'{path}\\error_N.png')
    print(f'File saved at {path}\\error_N.png')
    


        


def display_compare_norm(path: str, J: Function,
                         methodes_labels: tuple[tuple[METHOD_TYPE, str], ...],
                         X0: np.ndarray):
    '''
    Display the norm of the gradient of severals gradient descent methods at each iteration with the following parameters:
    - `path: str` the path to save the figure
    - `J: Function` a function object that has the `df` method
    - `methodes_labels: tuple[tuple[METHODE_TYPE, str], ...]` the methods to use with their labels
    - `X0: np.ndarray` the starting point of the method

    The function saves the figure at the path `path` and print a message to confirm the saving.
    '''
    plt.clf()
    for methode, label in methodes_labels:
        Xn = methode(J, X0, 5e-2, 1000)
    
        grad = [J.df(X) for X in Xn]

        plt.loglog(np.arange(len(Xn)), [np.linalg.norm(grad_val) for grad_val in grad], label=label)
    plt.legend()
    plt.grid()
    plt.xlabel('Nombre d\'itérations $i$')
    plt.ylabel(r'$\|\nabla f(x_i)\|$')
    plt.grid()

    plt.savefig(f'{path}\\convergence.png')
    print(f'File saved at {path}\\convergence.png')


def display_compare_error(path: str, J: Function,
                          methodes_labels: tuple[tuple[METHOD_TYPE, str], ...],
                          X0: np.ndarray, solution: np.ndarray):
    '''
    Display the error of severals gradient descent methods at each iteration with the following parameters:
    - `path: str` the path to save the figure
    - `J: Function` a function object that has the `df` method
    - `methodes_labels: tuple[tuple[METHODE_TYPE, str], ...]` the methods to use with their labels
    - `X0: np.ndarray` the starting point of the method
    - `solution: np.ndarray` the minimum of the function

    The function saves the figure at the path `path` and print a message to confirm the saving.
    '''
    plt.clf()
    for methode, label in methodes_labels:
        Xn = methode(J, X0, 5e-2, 20_000)
    
        err = np.array([np.linalg.norm(x - solution) for x in Xn])

        plt.loglog(np.arange(len(Xn)), err, label=label)
    plt.legend()
    plt.grid()
    plt.xlabel('Nombre d\'itérations $i$')
    plt.ylabel(r'$\|x_i-x^*\|$')
    plt.grid()
    plt.savefig(f'{path}\\error.png')
    print(f'File saved at {path}\\error.png')


def display_ka(path: str, nmax: int):
    """
    Display the condition number of the matrix A from the function `get_other_diago`.
    A will be a matrix of size 1 x 1 to nmax x nmax. The function takes the following parameters:
    - `path: str` the path to save the figure
    - `nmax: int` the maximum size of the matrix A

    The function saves the figure at the path `path` and print a message to confirm the saving.
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


def display_time_N(path: str, J_gen: Callable[[int], Function], methode: METHOD_TYPE, 
                   n_space: np.ndarray):
    """
    Display the time taken to compute the gradient descent method for a function of size n.
    The function takes the following parameters:
    - `path: str` the path to save the figure
    - `J_gen: Callable[[int], Function]` a function that generate a function of size n
    - `methode: METHODE_TYPE` the method to use
    - `n_space: np.ndarray` the values of n

    The function saves the figure at the path `path` and print a message to confirm the saving.
    """
    time = []
    for i in n_space:
        print(i)
        J = J_gen(i)
        X0 = np.array([[0] for _ in range(i)])
        timed_methode = time_func(methode)
        t, _ = timed_methode(J, X0, 5e-2, 1000)
        time.append(t)

    plt.clf()
    plt.title('Time taken to compute the gradient descent method')
    plt.plot(n_space, time)
    plt.grid()
    plt.savefig(f'{path}\\time_N.png')
    print(f'File saved at {path}\\time_N.png')



def display_phi(s, alphas):


    cmap = cm.get_cmap('Blues', len(alphas))

    i_colors = np.linspace(0, 1, len(alphas))
    i_max = len(alphas) - 1
    i_mean = len(alphas) // 2

    plt.clf()
    for i, (alpha, i_color) in enumerate(zip(alphas, i_colors)):

        y = computePhi(s, alpha)

        color = cmap(i_color)
        if i == i_mean:
            plt.plot(s, y, label=fr'$\phi_\alpha$', color=color)
        plt.plot(s, y, color=color)
    plt.plot(s, abs(s), ls='--', color='black', label=r'$|s|$')

    norm = Normalize(vmin=alphas[-1], vmax=alphas[0])
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), label=r'$\alpha$')

    plt.title(r'$\phi(s, \alpha)$ and $\|\phi(s)\|$')
    plt.xlabel('s')
    plt.ylabel(r'$\phi(s, \alpha)$')
    plt.legend()
    plt.savefig('figure/phi.svg')
    plt.show()

def estimate_u(img_np, lambda_):
    # res = minimize(X, img_np, args=(img_np, lambda_), method='CG', jac=grad_J, options={'maxiter': niter})
    # return res.x
    pass

def display_error_lambda(img_np, estimate_u):
    lambdas = np.linspace(0.1, 5, 50)  # Adjust as needed
    errors = []

    for lambda_ in lambdas:
        u_hat = estimate_u(img_np, lambda_)
        error = np.mean((img_np - u_hat)**2)
        errors.append(error)

    plt.plot(lambdas, errors)
    plt.xlabel('lambda')
    plt.ylabel('MSE')
    plt.title('MSE between original and estimated image vs lambda')
    plt.show()

