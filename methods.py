from typing import Callable
import numpy as np

from function import QuadraticFunction, Function



def quadratic_gradient_descent(f: QuadraticFunction, X0, eps: float, niter: int):
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
    return X, i

def quadratic_conjuguate_gradient_method(f: QuadraticFunction, X0, eps: float, niter: int):
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
    return X, i


def newton(f: Function, X0, eps: float, niter: int):
    i = 0
    X = np.array([X0])
    for i in range(niter):
        A = f.ddf(X[-1])
        b = f.df(X[-1])[:, 0]
        p = - np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1, 1)
        if np.linalg.norm(p) < eps:
            break
        X = np.append(X, np.array([X[-1] + p]), axis=0)
    return X, i

def gradient_descent(f: Function, X0, eps: float, niter: int):
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
    return X, i

def newton_optimal_step(f: Function, X0, eps: float, niter: int):
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
    return X, i
