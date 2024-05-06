import numpy as np


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
    :return:  QuadraticFunction object
    """
    A = a * np.eye(n)
    for i in range(n):
        A[i, i] = 3*(i+1)**2
    A = A - np.eye(n, k=1) - np.eye(n, k=-1)
    b = np.array([[i+1] for i in range(n)])
    c = 0
    return a*np.eye(n) + b*np.eye(n, k=1) + c*np.eye(n, k=-1)


def get_zvankin_quad(n):
    A = tri_diagonal(n, 2, -1, -1)
    b = np.array([[i+1] for i in range(n)])
    c = 0
    return QuadraticFunction(A, b, c)


def get_other_diago(n) :
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

