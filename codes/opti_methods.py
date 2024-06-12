from typing import Callable
import numpy as np
from tools import format_path
from function import (Function, grad_J, Jfonction)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy


METHOD_TYPE = Callable[[Function, np.ndarray, float, int],
                        np.ndarray]


def gradient_descent_fix_step(f: Function, X0: np.ndarray, eps: float, niter: int, alpha: float = 1e-2):
    """
    Find the minimum of a function using the gradient descent method with a fixed step with the following parameters :
    - `f: Function` a function object that has gradient method (`f.df`)
    - `X0: np.ndarray` the starting point of the method
    - `eps: float` the maximum error allowed
    - `niter: int` the maximum number of iterations allowed
    - `alpha: float` the step of the method

    The function returns an array of points `X: np.ndarray` which are the points of the path to the minimum
    and the last element of the array is the minimum of the function.
    """
    # Control that it's possible to find a minimum with the given step
    if alpha >= eps:
        raise ValueError('alpha must be less than eps')
    X = np.array([X0])
    for i in range(niter):
        grad = f.df(X[-1])
        norm = np.linalg.norm(grad)
        p = - grad / norm
        if norm < eps:
            break
        X = np.append(X, np.array([X[-1] + alpha * p]), axis=0)
    return X

def quadratic_gradient_descent_optimal_step(f: Function, X0: np.ndarray, eps: float, niter: int):
    """
    Return the minimum of a function with the postulat that the function is quadratic
    using the gradient descent method with an optimal step with the following parameters :
    - `f: Function` a function object that has gradient and hessian methods (`f.df` and `f.ddf`)
    - `X0: np.ndarray` the starting point of the method
    - `eps: float` the maximum error allowed
    - `niter: int` the maximum number of iterations allowed

    The function returns an array of points `X: np.ndarray` which are the points of the path to the minimum
    and the last element of the array is the minimum of the function.
    """
    X = np.array([X0])
    for i in range(niter):
        r = p = - f.df(X[-1])
        if np.linalg.norm(r) < eps:
            break
        alpha = (r.T @ r)/(r.T @ f.ddf(X[-1]) @ r)
        X = np.append(X, np.array([X[-1] + alpha * p]), axis=0)
    return X

def quadratic_conjuguate_gradient(f: Function, X0: np.ndarray, eps: float, niter: int):
    """
    Return the minimum of a function with the postulat that the function is quadratic
    using the conjuguate gradient method with the following parameters :
    - `f: Function` a function object that has gradient and hessian methods (`f.df` and `f.ddf`)
    - `X0: np.ndarray` the starting point of the method
    - `eps: float` the maximum error allowed
    - `niter: int` the maximum number of iterations allowed

    The function returns an array of points `X: np.ndarray` which are the points of the path to the minimum
    and the last element of the array is the minimum of the function.
    """

    X = np.array([X0])
    r0 = p = - f.df(X0) # equivalent to r0 = p = b - A @ X0
    for i in range(niter):
        alpha = (r0.T @ r0)/(p.T @ f.ddf(X[-1]) @ p) # equivalent to alpha = (r0.T @ r0)/(p.T @ A @ p)
        X = np.append(X, np.array([X[-1] + alpha * p]), axis=0)
        r1 = - f.df(X[-1])
        if np.linalg.norm(r1) < eps:
            break
        beta = (r1.T @ r1)/(r0.T @ r0)
        p = r1 + beta * p
        r0 = r1
    return X

def newton(f: Function, X0: np.ndarray, eps: float, niter: int):
    """
    Find the minimum of a function using the newton method with the following parameters :
    - `f: Function` a function object that has gradient and hessian methods (`f.df` and `f.ddf`)
    - `X0: np.ndarray` the starting point of the method
    - `eps: float` the maximum error allowed
    - `niter: int` the maximum number of iterations allowed

    The function returns an array of points `X: np.ndarray` which are the points of the path to the minimum
    and the last element of the array is the minimum of the function.
    """
    X = np.array([X0])
    for i in range(niter):
        A = f.ddf(X[-1])
        b = f.df(X[-1])[:, 0]
        p = - np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1, 1)
        if np.linalg.norm(p) < eps:
            break
        X = np.append(X, np.array([X[-1] + p]), axis=0)
    return X

def gradient_descent_optimal_step(f: Function, X0: np.ndarray, eps: float, niter: int):
    """
    Find the minimum of a function using the gradient descent method with an optimal step
    that is found using the newton method. The method use the following parameters :
    - `f: Function` a function object that has gradient and hessian methods (`f.df` and `f.ddf`)
    - `X0: np.ndarray` the starting point of the method
    - `eps: float` the maximum error allowed
    - `niter: int` the maximum number of iterations allowed

    The function returns an array of points `X: np.ndarray` which are the points of the path to the minimum
    and the last element of the array is the minimum of the function.
    """
    # !!!!! ATTENTION MARCHE PAS !!!!!
    X = np.array([X0])
    for i in range(niter):
        p = - f.df(X[-1])
        if np.linalg.norm(p) < eps:
            break
        f_alpha = f.partial(X[-1], p)
        alpha = newton(f_alpha, np.array([[0]]), eps, niter)
        alpha = alpha[-1]
        X = np.append(X, np.array([X[-1] + alpha * p]), axis=0)
    return X

def newton_optimal_step(f: Function, X0: np.ndarray, eps: float, niter: int):
    """
    Find the minimum of a function using the newton method with an optimal step
    that is found using the newton method. The method use the following parameters :
    - `f: Function` a function object that has gradient and hessian methods (`f.df` and `f.ddf`)
    - `X0: np.ndarray` the starting point of the method
    - `eps: float` the maximum error allowed
    - `niter: int` the maximum number of iterations allowed

    The function returns an array of points `X: np.ndarray` which are the points of the path to the minimum
    and the last element of the array is the minimum of the function.
    """
    X = np.array([X0])
    for i in range(niter):
        A = f.ddf(X[-1])
        b = f.df(X[-1])[:, 0]
        p = - np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1, 1)
        if np.linalg.norm(p) < eps:
            break
        f_alpha = f.partial(X[-1], p)
        alpha = newton(f_alpha, np.array([[0]]), eps, niter)
        alpha = alpha[-1]
        X = np.append(X, np.array([X[-1] + alpha * p]), axis=0)
    return X

def quasi_newton(f: Function,x0: np.ndarray,eps: float,max_iter: int, method='BFGS'):
    N=len(x0)
    B=np.eye(N)  # Initial B matrix (Identity)
    X=[x0]

    while np.linalg.norm(f.df(X[-1]))>eps and len(X)<max_iter:
        grad=f.df(X[-1])
        d=-np.dot(B,grad)

        # Minimize along the search direction
        alpha_fun=lambda nu: f.f(X[-1]+nu*d)
        res=scipy.optimize.minimize_scalar(alpha_fun)
        nu=res.x
        X.append(X[-1]+nu*d)

        s=X[-1]-X[-2]
        y=f.df(X[-1])-f.df(X[-2])

        if method == 'BFGS':
            Bs=np.dot(B,s)
            sy=np.dot(s.T,y)
            B=B+(np.outer(y,y.T)/sy)-(np.outer(Bs,Bs.T)/np.dot(s.T,Bs))
        elif method == 'DFP':
            sy=np.dot(s.T,y)
            y_B=np.dot(B,y)
            B=B+(np.outer(s,s.T)/sy)-(np.outer(y_B,y_B.T)/np.dot(y.T,y_B))

    return np.array(X)

def quasi_newton_BFGS(f: Function, X0: np.ndarray, eps: float, niter: int):
    return quasi_newton(f, X0, eps, niter, method='BFGS')

def quasi_newton_DFP(f: Function, X0: np.ndarray, eps: float, niter: int):
    return quasi_newton(f, X0, eps, niter, method='DFP')

def gradien_pas_fixe_J(v, u0, nb_iter, pas, lambda_, alpha):
    u = u0.copy()
    for _ in range(nb_iter):
        grad = grad_J(u, v, lambda_, alpha)
        u = u - pas * grad
    return u


import numpy as np
import scipy.optimize

def dfp_J(v, u0, nb_iter, lambda_, alpha, eps=1e-3):
    u = u0.copy()
    m, n = u.shape
    N = m * n
    B = np.eye(N)  # Initial B matrix (Identity)
    u_flat = u.flatten()

    def grad_J_flat(u_flat):
        u = u_flat.reshape((m, n))
        return grad_J(u, v, lambda_, alpha).flatten()

    def Jfonction_flat(u_flat):
        u = u_flat.reshape((m, n))
        return Jfonction(u, v, lambda_, alpha)

    iter_count = 0
    while np.linalg.norm(grad_J_flat(u_flat)) > eps and iter_count < nb_iter:
        grad = grad_J_flat(u_flat)
        d = -np.dot(B, grad)

        # Minimize along the search direction
        alpha_fun = lambda nu: Jfonction_flat(u_flat + nu * d)
        res = scipy.optimize.minimize_scalar(alpha_fun)
        nu = res.x

        s = nu * d
        u_flat = u_flat + s

        grad_new = grad_J_flat(u_flat)
        y = grad_new - grad

        if np.linalg.norm(y) < 1e-5:
            break

        sy = np.dot(s.T, y)
        y_B = np.dot(B, y)
        B = B + np.dot(s, s.T) / sy - np.dot(np.dot(B,y), np.dot(y.T,B)) / np.dot(y.T, y_B)

        iter_count += 1

    u = u_flat.reshape((m, n))
    return u

def BFGS_J(v, u0, nb_iter, lambda_, alpha, eps=1e-3):
    u = u0.copy()
    m, n = u.shape
    N = m * n
    B = np.eye(N)  # Initial B matrix (Identity)
    u_flat = u.flatten()

    def grad_J_flat(u_flat):
        u = u_flat.reshape((m, n))
        return grad_J(u, v, lambda_, alpha).flatten()

    def Jfonction_flat(u_flat):
        u = u_flat.reshape((m, n))
        return Jfonction(u, v, lambda_, alpha)

    iter_count = 0
    while np.linalg.norm(grad_J_flat(u_flat)) > eps and iter_count < nb_iter:
        grad = grad_J_flat(u_flat)
        d = -np.dot(B, grad)

        # Minimize along the search direction
        alpha_fun = lambda nu: Jfonction_flat(u_flat + nu * d)
        res = scipy.optimize.minimize_scalar(alpha_fun)
        nu = res.x

        s = nu * d
        u_flat = u_flat + s

        grad_new = grad_J_flat(u_flat)
        y = grad_new - grad

        if np.linalg.norm(y) < 1e-5:
            break

        Bs = np.dot(B,s)
        ys = np.dot(y.T,s)
        B  = B + (np.dot(y,y.T)/ys )-(np.dot(Bs,np.dot(s.T,B))/np.dot(s.T,Bs))

        iter_count += 1

    u = u_flat.reshape((m, n))
    return u


__METHODS_LABEL: dict[METHOD_TYPE, str] = {
    quadratic_gradient_descent_optimal_step : 'Quadratic Gradient Descent Optimal Step',
    quadratic_conjuguate_gradient           : 'Conjuguate Gradient',
    gradient_descent_fix_step               : 'Gradient Descent Fixed Step',
    gradient_descent_optimal_step           : 'Gradient Descent Optimal Step',
    newton                                  : 'Newton',
    newton_optimal_step                     : 'Newton Optimal Step',
    quasi_newton_BFGS                       : 'Quasi Newton BFGS',
    quasi_newton_DFP                        : 'Quasi Newton DFP',
}

METHODS_LABEL_PATH: dict[METHOD_TYPE, tuple[str, str]] = {method: (label, format_path(label)) for method, label in __METHODS_LABEL.items()}

