import numpy as np


class Function:
    def __init__(self, f, df, ddf):
        self.f = f
        self.df = df
        self.ddf = ddf

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
    """
    create a tri-diagonal matrix of size n
    with 2 on the diagonal and -1 on the sub-diagonal and super-diagonal
    :param n: size of the matrix
    :return: quadratic function object
    """
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
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


def condi_A(f: QuadraticFunction) :
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

    def grad(X):
        return np.array([[np.sum([A[k, i] * np.exp(A[:, i].T @ X + b[i]) for i in range(A.shape[1])])]
                         for k in range(A.shape[0])])

    def hessian(X):
        return np.array([[np.sum([A[k, i] * A[l, i] * np.exp(A[:, i].T @ X + b[i]) for i in range(A.shape[1])])
                         for l in range(A.shape[0])]
                         for k in range(A.shape[0])])

    return Function(func, grad, hessian)

def get_J_2(n: int):
    def func(X):
        return np.sum([np.log(np.exp(X[i])) for i in range(n)])

    def grad(X):
        return np.array([[np.exp(X[i]) / np.sum([np.exp(X[j]) for j in range(n)])] for i in range(n)])

    def hessian(X):
        sum_exp = np.sum([np.exp(X[j]) for j in range(n)])
        sum_exp_square = sum_exp**2
        return np.array([[-np.exp(X[i]+X[j]) / sum_exp_square
                         for j in range(n)]
                         for i in range(n)]) + np.diag([np.exp(X[i]) / sum_exp for i in range(n)])

    return Function(func, grad, hessian)


def gen_J_1(n: int):
    A = np.random.randn(n, n)
    b = np.array([[-0.1] for _ in range(n)])
    return get_J_1(A, b)


def computePhi(s, alpha):
    return abs(s) - alpha * np.log((alpha + abs(s)) / alpha)
def computeAbsPhi(s, alpha):
    return abs( abs(s) - alpha * np.log(1 + abs(s)/alpha))

def dphi(s, alpha):
    """
    compute the derivative of the function phi_alpha
    - `s: np.ndarray` the input of the function
    - `alpha: float` the parameter of the function
    return the derivative of the function phi_alpha
    """
    return np.where(s>=0,s/(alpha+s),s/(alpha-s))

def Jfonction (u, v, lmbd, alpha):
    gx, gy = np.gradient(u)
    return 0.5 * np.linalg.norm(u - v)**2 + lmbd * (np.sum(computePhi(gx, alpha)) + np.sum(computePhi(gy, alpha)))

def grad_J(u , v, lmbd, alpha):
    gx = np.gradient(u, axis=1)
    gy = np.gradient(u, axis=0)
    divergence = np.gradient(dphi(gx, alpha), axis=1) + np.gradient(dphi(gy, alpha), axis=0)
    return u - v - lmbd * divergence


if __name__ == '__main__':

    Quad_func = get_zvankin_quad(2)

    print("========= TEST Zvankin quadratic func =======================")
    print(f"{Quad_func.A = }")
    print(f"{Quad_func.b = }")
    print(f"{Quad_func.c = }")


    A = np.array([[1.0, -1.0],
                  [3.0,  0.0]])
    b = np.array([[-0.1],
                  [-0.1]])

    J1 = get_J_1(A, b)

    J2 = get_J_2(2)

    print()
    print("========= TEST get_J_1 =======================")
    X0 = np.array([[-5.0], [5.0]])
    f_J1 = 20_064.66
    df_J1 = np.array([[19_796.08],
                      [59_791.11]])
    ddf_J1 = np.array([[20_064.66,  59_791.11],
                       [59_791.11, 179_373.33]])
    print(f"{A = }")
    print(f"{b = }")
    print(f"{X0 = }")
    print(f"{J1(X0) = }")
    print(f"{J1.df(X0) = }")
    print(f"{J1.ddf(X0) = }")
    print(f"{J1(X0) - f_J1 = }")
    print(f"{J1.df(X0) - df_J1 = }")
    print(f"{J1.ddf(X0) - ddf_J1 = }")

    print()
    print("========= TEST get_J_2 =======================")
    x, y = 1.0, 2.0
    X0 = np.array([[x], [y]])
    f_J2 = np.log(np.exp(x) + np.exp(y))
    df_J2 = (1/(np.exp(x) + np.exp(y))) * np.array([[np.exp(x)], [np.exp(y)]])
    ddf_J2 = (np.exp(x+y)/(np.exp(x) + np.exp(y))**2) * np.array([[1, -1],
                                                                  [-1,  1]])
    print(f"{X0 = }")
    print(f"{J2(X0) = }")
    print(f"{J2.df(X0) = }")
    print(f"{J2.ddf(X0) = }")
    print(f"{J2(X0) - f_J2 = }")
    print(f"{J2.df(X0) - df_J2 = }")
    print(f"{J2.ddf(X0) - ddf_J2 = }")

