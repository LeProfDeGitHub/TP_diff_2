from matplotlib import pyplot as plt
import numpy as np
from display_TP1 import plot_contour
from opti_methods import METHODE_TYPE
from function import Function

# def display_convergence_2d(path: str, f: Function, X0: np.ndarray, methode: METHODE_TYPE):
#     Xn = methode(f, X0, 5e-2, 1000)
    
#     plot_contour(f, (-10, 10), (-10, 10))
#     plt.plot(Xn[:, 0], Xn[:, 1], 'r*--', label='Gradient Descente')
#     plt.savefig(f'{path}\\convergence_grad_de.png')
#     print(f'File saved at {path}\\convergence_grad_de.png')