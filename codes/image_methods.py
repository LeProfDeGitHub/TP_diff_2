import numpy as np


def modify_image(image_np):
    """
    Modify the mean and standard deviation of an image
    :param Im: Image to be modified
    """
    image_np = image_np.astype(np.float32)
    image_np = (image_np - np.mean(image_np)) / np.std(image_np)
    return image_np



def grad(image_np) -> tuple[np.ndarray, np.ndarray]:
    """
    Affiche le gradient de l'image.
    :param Im: Image à traiter
    :return: gradient de l'image
    """
    m,n=image_np.shape

    gradient_x=np.zeros((m,n))
    gradient_y=np.zeros((m,n))

    gradient_x[:-1,:]=image_np[1:,:]-image_np[:-1,:]
    gradient_x[-1,:]=0

    gradient_y[:,:-1]=image_np[:,1:]-image_np[:,:-1]
    gradient_y[:,-1]=0

    return gradient_x, gradient_y

def grad_norm(u):
    """
    Retourne une image N qui correspond à la norme du gradient de l'image u. 
    :param Im: Image à traiter
    :return: Image retournée N
    """
    image_np = u.astype(np.float32)

    # Calcul du gradient
    gx, gy = grad(image_np)
    imgrad_np = np.sqrt(gx**2 + gy**2)

    return imgrad_np

# def div(p):
#     """
#     Calcule la divergence d'un champ de vecteurs p = (p1, p2) de dimension (m, n).
    
#     Arguments:
#     p -- tuple (p1, p2), où p1 et p2 sont des arrays numpy de dimension (m, n)
    
#     Retourne:
#     divergence -- array numpy de dimension (m, n)
#     """
#     p1, p2 = p
#     print(p1.shape, p2.shape)
#     m, n = p1.shape
    
#     # Calcul de ∂*x p1
#     D_partielle_x_p1 = np.zeros((m, n))
#     D_partielle_x_p1[0, :] = p1[0, :]
#     D_partielle_x_p1[1:m-1, :] = p1[1:m-1, :] - p1[0:m-2, :]
#     D_partielle_x_p1[m-1, :] = -p1[m-2, :]
    
#     # Calcul de ∂*y p2
#     D_partielle_y_p2 = np.zeros((m, n))
#     D_partielle_y_p2[:, 0] = p2[:, 0]
#     D_partielle_y_p2[:, 1:n-1] = p2[:, 1:n-1] - p2[:, 0:n-2]
#     D_partielle_y_p2[:, n-1] = -p2[:, n-2]
    
#     # Calcul de la divergence
#     divergence = D_partielle_x_p1 + D_partielle_y_p2
    
#     return divergence

def div(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calcule la divergence d'un champ de vecteurs v = (v1, v2) de dimension (m, n).
    - `v` est un array numpy de dimension (m, n)

    """
    m, n = v1.shape

    div_omega = np.zeros((m, n))

    div_omega[:-1, :] += v1[:-1, :]
    div_omega[1:, :] -= v1[:-1, :]

    div_omega[:, :-1] += v2[:, :-1]
    div_omega[:, 1:] -= v2[:, :-1]

    return div_omega


def test_div_grad():
    """
    Test la divergence et le gradient de deux vecteurs u et v
    avec la relation <grad(u),v> = -<u,div(v)>
    :return: booléen
    """

    u = np.random.rand(10, 10)
    omega_x = np.random.rand(10, 10)
    omega_y = np.random.rand(10, 10)

    grad_u_x, grad_u_y = grad(u)
    div_omega = div(omega_x, omega_y)

    inner_product_grad = np.sum(grad_u_x * omega_x + grad_u_y * omega_y)
    inner_product_div = np.sum(u * div_omega)

    print("grad(u) * v:", inner_product_grad)
    print("u * div(v):", inner_product_div)
    print("Test passed:", np.abs(inner_product_grad + inner_product_div) < 1e-10)



def add_noise(image_np, sigma):
    """
    Ajoute du bruit gaussien à une image.
    :param Im: Image à traiter
    :param sigma: Ecart-type du bruit
    :return: Image bruitée
    """

    image_np = image_np.astype(np.float32)
    noise = np.random.normal(0, sigma, image_np.shape)
    image_noisy = image_np + noise
    
    return image_noisy








