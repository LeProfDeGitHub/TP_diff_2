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

    return gradient_x,gradient_y
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

def div(v: np.ndarray) -> np.ndarray:
    """
    Calcule la divergence d'un champ de vecteurs v = (v1, v2) de dimension (m, n).
    - `v` est un array numpy de dimension (m, n)

    """
    m, n = v.shape
    divv = np.zeros((m, n))

    # Calcul de la divergence selon les définitions données
    divv[1:-1, :] = v[1:-1, :] - v[:-2, :]
    divv[:, 1:-1] += v[:, 1:-1] - v[:, :-2]
    divv[0, :] += v[0, :]
    divv[:, 0] += v[:, 0]
    divv[-1, :] -= v[-2, :]
    divv[:, -1] -= v[:, -2]

    return divv
def test_div_grad(u,v):
    """
    Test la divergence et le gradient du vecteur u et v
    avec la relation <grad(u),v> = -<u,div(v)>
    :param u: vecteur u
    :param v: vecteur v
    :return: booléen
    """

    div_v=div(v)
    grad_u=grad(u)

    print("div(v):",div_v)
    print("grad(u):",grad_u)

    u_div_v=np.sum(u*div_v)
    grad_u_v=np.sum(grad_u*v)
    test=np.allclose(grad_u_v,-u_div_v)

    print("grad(u) * v:",grad_u_v)
    print("u * div(v):",u_div_v)
    print("Test passed:",test)
    return test


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








