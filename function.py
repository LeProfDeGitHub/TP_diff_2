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





def get_zvankin_quad(n):
    A = 2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    b = np.array([[i+1] for i in range(n)])
    c = 0
    return QuadraticFunction(A, b, c)


