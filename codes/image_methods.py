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
    image_np = image_np.astype(np.float32)
    # Calcul du gradient
    gx, gy = np.gradient(image_np)
    
    return gx, gy

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

def div(image_np):
    u, v = grad(image_np)
    return u + v

def test_div_grad(u, v) :
    """
    test la divergence et le gradient du vecteur u et v
    avec la relation <grad(u),v> = -<u,div(v)>
    :param u: vecteur u
    :param v: vecteur v
    :return: booléen
    """
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    # Calcul du gradient
    gx, gy = grad(u)
    dx, dy = div(v)
    test = np.allclose(np.sum(gx*v + gy*v), -np.sum(u*dx + u*dy))
    print(test)
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








