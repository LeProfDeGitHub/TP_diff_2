import autograd
import autograd.numpy as np


class Function:
    def __init__(self, f, df, ddf):
        self.f = f
        self.df = df
        self.ddf = ddf

    def f(self, X):
        return self.f(X)

    def df(self, X):
        return self.df(X)

    def ddf(self, X):
        return self.ddf(X)

    def __call__(self, X):
        return self.f(X)

    def partial(self, X, p):
        return Function(lambda alpha: self.f(X + alpha * p),
                        lambda alpha: self.df(X + alpha * p).T @ p,
                        lambda alpha: p.T @ self.ddf(X + alpha * p) @ p)


class QuadraticFunction(Function):
    def __init__(self, A, b, c):
        self.f   = lambda X: 0.5 * X.T @ A @ X - b.T @ X + c
        self.df  = lambda X: A @ X - b
        self.ddf = lambda X: A

        self.A = A
        self.b = b
        self.c = c


def tri_diagonal(n, a, b, c) :
    """
    create a tri-diagonal matrix of size n
    with 3*i**2 on the diagonal and -1 on the sub-diagonal and super-diagonal
    :param n: size of the matrix
    :return:  Matrix A of size n
    """
    A = a * np.eye(n)
    for i in range(n):
        A[i, i] = 3*(i+1)**2
    A = A - np.eye(n, k=1) - np.eye(n, k=-1)
    b = np.array([[i+1] for i in range(n)])
    c = 0
    return a*np.eye(n) + b*np.eye(n, k=1) + c*np.eye(n, k=-1)


def get_zvankin_quad(n):
    """
    create a tri-diagonal matrix of size n
    with 2 on the diagonal and -1 on the sub-diagonal and super-diagonal
    :param n: size of the matrix
    :return: quadratic function object
    """
    A = tri_diagonal(n, 2, -1, -1)
    b = np.array([[i+1] for i in range(n)])
    c = 0
    return QuadraticFunction(A, b, c)


def get_other_diago(n: int):
    """
    create a tri-diagonal matrix of size n
    with 3*i**2 on the diagonal and -1 on the sub-diagonal and super-diagonal
    :param n: size of the matrix
    :return:  QuadraticFunction object
    """
    A = np.eye(n)
    for i in range(n):
        A[i, i] = 3*(i+1)**2
    A = A - np.eye(n, k=1) - np.eye(n, k=-1)
    b = np.array([[i+1] for i in range(n)])
    c = 0
    return QuadraticFunction(A, b, c)


def condi_A( f : QuadraticFunction) :
    """
    condition A with the diagonal matrix D = diag(1/(i+1))
    :param f: QuadraticFunction
    :return: QuadraticFunction with A conditioned
    """
    D = np.zeros((f.A.shape[0], f.A.shape[1]))
    for i in range(f.A.shape[0]):
        D[i, i] = 1/(i+1)
    A = D @ f.A @ D
    b = D @ f.b
    c = f.c
    return QuadraticFunction(A, b, c)


def get_J_1(A: np.ndarray, b: np.ndarray):
    def func(X):
        return np.sum([np.exp(A[:, i].T @ X + b[i]) for i in range(A.shape[1])])

    return Function(func, autograd.grad(func), autograd.hessian(func))

def get_J_2(n: int):
    def func(X):
        return np.sum([np.log(np.exp(X[i])) for i in range(n)])

    return Function(func, autograd.grad(func), autograd.hessian(func))

