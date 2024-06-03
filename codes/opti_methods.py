from typing import Callable
import numpy as np
from tools import format_path
from function import Function


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

def quadratic_conjuguate_gradient_method(f: Function, X0: np.ndarray, eps: float, niter: int):
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

def BFGS(f: Function, x0: np.ndarray, eps: float, n: int):
    """
    Find the minimum of a function using the BFGS method with the following parameters :
    - `f: Function` a function object that has gradient method (`J.df`)
    - `x0: np.ndarray` the starting point of the method
    - `eps: float` the maximum error allowed
    - `n: int` the maximum number of iterations allowed

    The function returns an array of points `X: np.ndarray` which are the points of the path to the minimum
    and the last element of the array is the minimum of the function.
    """
    X = np.array([x0])
    B = np.eye(len(x0))
    while np.linalg.norm(f.df(X[-1])) > eps and len(X) < n :
        d = - B @ f.df(X[-1])
        alpha = BFGS(f.partial(X[-1], d), np.array([[0]]), eps, n)[0]
        alpha = alpha[-1]
        x = X[-1] + alpha * d
        s = x - X[-1]
        y = f.df(x) - f.df(X[-1])
        B = B + (y @ y.T)/(y.T @ s) - (B @ s @ s.T @ B)/(s.T @ B @ s)
        X = np.append(X, np.array([x]), axis=0)
    return X


__METHODS_LABEL: dict[METHOD_TYPE, str] = {
    gradient_descent_fix_step              : 'Gradient Descent Fixed Step',
    quadratic_gradient_descent_optimal_step: 'Quadratic Gradient Descent Optimal Step',
    quadratic_conjuguate_gradient_method   : 'Conjuguate Gradient',
    newton                                 : 'Newton',
    gradient_descent_optimal_step          : 'Gradient Descent Optimal Step',
    newton_optimal_step                    : 'Newton Optimal Step',
    BFGS                                   : 'BFGS',
}

METHODS_LABEL_PATH: dict[METHOD_TYPE, tuple[str, str]] = {method: (label, format_path(label)) for method, label in __METHODS_LABEL.items()}
