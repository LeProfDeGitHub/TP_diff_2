from matplotlib import pyplot as plt
import numpy as np

from function import Function
from matplotlib import pyplot as plt
from matplotlib import colors as plt_color



def plot_contour(f: Function, xlim: tuple[float, float], ylim: tuple[float, float], norm=None):
    x, y = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 100),
        np.linspace(ylim[0], ylim[1], 100)
    )

    z = np.array([f(np.array([[x[i, j]], [y[i, j]]]))[0] for i in range(100) for j in range(100)]).reshape(100, 100)

    k = np.linspace(np.nanmin(z), np.nanmax(z), 100)

    contour = plt.contourf(x, y, z, levels=k, cmap='viridis', alpha=1, norm=norm)
    
    plt.colorbar(contour, label='f(x, y)')
