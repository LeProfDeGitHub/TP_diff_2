from typing import Callable, NewType
import numpy as np
from function import QuadraticFunction, Function, condi_A

METHODE_TYPE = Callable[[QuadraticFunction, np.ndarray, float, int], tuple[np.ndarray, int]]

def quadratic_gradient_descent(f: QuadraticFunction, X0, eps: float, niter: int):
    """
    find the minimum of a quadratic function using the gradient descent method
    :param f: quadratic function object
    :param X0: starting point
    :param eps: error
    :param niter: number of iterations
    :return: X : array of points, i+1 : number of iterations
    """
    i = 0
    X = np.array([X0])
    for i in range(niter):
        r = p = f.b - f.A @ X[-1]
        if np.linalg.norm(r) < eps:
            break
        alpha = (r.T @ r)/(r.T @ f.A @ r)
        old_x = X[-1]
        new_x = old_x + alpha * p
        X = np.append(X, np.array([X[-1] + alpha * p]), axis=0)
    return X, i+1

def quadratic_conjuguate_gradient_method(f: QuadraticFunction, X0, eps: float, niter: int):
    """
    find the minimum of a quadratic function using the conjuguate gradient method
    :param f: quadratic function object
    :param X0: starting point
    :param eps: error
    :param niter: number of iterations
    :return: X : array of points, i+1 : number of iterations
    """
    i = 0
    X = np.array([X0])
    r0 = p = f.b - f.A @ X0
    for i in range(niter):
        alpha = (p.T @ r0)/(p.T @ f.A @ p)
        X = np.append(X, np.array([X[-1] + alpha * p]), axis=0)
        r1 = f.b - f.A @ X[-1]
        if np.linalg.norm(r1) < eps:
            break
        beta = (r1.T @ r1)/(r0.T @ r0)
        p = r1 + beta * p
        r0 = r1
    return X, i+1


def newton(f: Function, X0, eps: float, niter: int):
    """
    find the minimum of a function using the newton method
    :param f: function object
    :param X0: starting point
    :param eps: error
    :param niter: number of iterations
    :return: X : array of points, i+1 : number of iterations
    """
    i = 0
    X = np.array([X0])
    for i in range(niter):
        A = f.ddf(X[-1])
        b = f.df(X[-1])[:, 0]
        p = - np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1, 1)
        if np.linalg.norm(p) < eps:
            break
        X = np.append(X, np.array([X[-1] + p]), axis=0)
    return X, i+1

def gradient_descent_fix_step(f: Function, X0, eps: float, niter: int, alpha: float = 1e-3):
    """
    use the gradient descent method with a fixed step to find the minimum of a function
    :param f: function object
    :param X0: starting point
    :param eps: error
    :param niter: number of iterations
    :param alpha: step used in the gradient descent method
    :return: X : array of points, i+1 : number of iterations
    """
    i = 0
    X = np.array([X0])
    for i in range(niter):
        p = - f.df(X[-1])
        if np.linalg.norm(p) < eps:
            break
        X = np.append(X, np.array([X[-1] + alpha * p]), axis=0)
    return X, i+1

def gradient_descent_optimal_step(f: Function, X0, eps: float, niter: int):
    """
    use the gradient descent method with an optimal step to find the minimum of a function
    :param f: fonction object
    :param X0: starting point
    :param eps: error
    :param niter: number of iterations
    :return: X : array of points, i+1 : number of iterations
    """
    i = 0
    X = np.array([X0])
    for i in range(niter):
        p = - f.df(X[-1])
        if np.linalg.norm(p) < eps:
            break
        f_alpha = f.partial(X[-1], p)
        alpha, _ = newton(f_alpha, np.array([[0]]), eps, niter)
        alpha = alpha[-1]
        X = np.append(X, np.array([X[-1] + alpha * p]), axis=0)
    return X, i+1

def newton_optimal_step(f: Function, X0, eps: float, niter: int):
    """
    find the minimum of a function using the newton method with an optimal step
    :param f: function object
    :param X0: starting point
    :param eps: error
    :param niter: number of iterations
    :return: X : array of points, i+1 : number of iterations
    """
    i = 0
    X = np.array([X0])
    for i in range(niter):
        A = f.ddf(X[-1])
        b = f.df(X[-1])[:, 0]
        p = - np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1, 1)
        if np.linalg.norm(p) < eps:
            break
        f_alpha = f.partial(X[-1], p)
        alpha, _ = newton(f_alpha, np.array([[0]]), eps, niter)
        alpha = alpha[-1]
        X = np.append(X, np.array([X[-1] + alpha * p]), axis=0)
    return X, i+1


def comparaison_condi(f : QuadraticFunction, X0, eps: float, niter: int):
    """
    compare the number of iterations of the gradient descent method
    and the conjugate gradient method for a given quadratic function.

    :param f: QuadraticFunction object
    :param X0: starting point
    :param eps: error
    :param niter: number of iterations
    :return: none
    """
    print("------------------------------------------")
    print("           unconditioned matrix           ")
    print("------------------------------------------\n")
    X_gd, i_max = quadratic_gradient_descent(f, X0, eps, niter)
    print(f'gradient descent method optimal step: {i_max} iterations')
    error = np.linalg.norm(f.df (X_gd[-1]))
    print(f'error: {error}\n')

    X_cg, i_max = quadratic_conjuguate_gradient_method(f, X0, eps, niter)
    print(f'conjuguate gradient method: {i_max} iterations')
    error = np.linalg.norm(f.df (X_cg[-1]))
    print(f'error: {error }\n')

    print("------------------------------------------")
    print("             conditioned matrix           ")
    print("------------------------------------------\n")

    quad = condi_A(f)
    X_gd, i_max = quadratic_gradient_descent(quad, X0, eps, niter)
    print(f'gradient descent method optimal step: {i_max} iterations')
    error = np.linalg.norm(quad.df (X_gd[-1]))
    print(f'error: {error }\n')

    X_cg, i_max = quadratic_conjuguate_gradient_method(quad, X0, eps, niter)
    print(f'conjuguate gradient method: {i_max} iterations')
    error = np.linalg.norm(quad.df (X_cg[-1]))
    print(f'error: {error}\n')

    return None

def BFGS(J : Function , x0 , eps : float, n :int) :
    """
    find the minimum of a function using the
    Broyden Fletcher Goldfarb Shanno method
    :param J: function object
    :param x0: starting point
    :param eps: error
    :param n: number of iterations
    :return: X : array of points, i+1 : number of iterations
    """
    X = np.array([x0])
    B = np.eye(len(x0))
    while np.linalg.norm(J.df(X[-1])) > eps and len(X) < n :
        d = - B @ J.df(X[-1])
        alpha = BFGS(J.partial(X[-1], d), np.array([[0]]), eps, n)[0]
        alpha = alpha[-1]
        x = X[-1] + alpha * d
        s = x - X[-1]
        y = J.df(x) - J.df(X[-1])
        B = B + (y @ y.T)/(y.T @ s) - (B @ s @ s.T @ B)/(s.T @ B @ s)
        X = np.append(X, np.array([x]), axis=0)
    return X, len(X)