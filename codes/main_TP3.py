import PIL.Image
import numpy as np 
import PIL
import matplotlib.pyplot as plt

from function import Function
from image_methods import (modify_image,
                           grad,
                           grad_norm, div,
                           test_div_grad,
                           add_noise,)
from opti_methods import (METHOD_TYPE,
                          METHODS_LABEL_PATH,
                          gradient_descent_fix_step,
                          quadratic_gradient_descent_optimal_step,
                          quadratic_conjuguate_gradient,
                          newton,
                          newton_optimal_step,
                          gradient_descent_optimal_step)
from display import display_phi


METHODS: tuple[METHOD_TYPE, ...] = (
    gradient_descent_fix_step,
    quadratic_gradient_descent_optimal_step,
    quadratic_conjuguate_gradient,
    newton,
    newton_optimal_step,
    gradient_descent_optimal_step,
    )


def test_hist(img):
    
    img_np = np.array(img)
    hist5 ,bin_edges5= np.histogram(img_np, bins=5, range=(0, 5))
    hist256 , bin_edges256 = np.histogram(img_np, bins=256, range=(0, 256))    
    plt.clf()
    plt.bar(bin_edges5[:-1], hist5, width=1)
    plt.xlabel('Gray level')
    plt.ylabel('pixel count')
    plt.title('Histogram with 5 Bins')
    plt.show()
    plt.clf()
    plt.bar(bin_edges256[:-1], hist256, width=1)
    plt.xlabel('Gray level')
    plt.ylabel('pixel count')
    plt.title('Histogram with 256 Bins')
    plt.show()
    




def test_reshape(img):
    img.resize((100,100)).show()
    img.rotate(45).show()

def test_modifie_image(img):
    new_img_np = modify_image(np.array(img))
    new_img = PIL.Image.fromarray(new_img_np.astype(np.uint8))
    new_img.show()

def test_grad(img):
    gx, gy = grad(np.array(img))
    gx_img = PIL.Image.fromarray(gx.astype(np.uint8))
    gy_img = PIL.Image.fromarray(gy.astype(np.uint8))
    gxy_img = PIL.Image.fromarray(grad_norm(np.array(img)).astype(np.uint8))
    gx_img.show()
    gy_img.show()
    gxy_img.show()

def test_div(img):
    div_img = div(np.array(img))
    div_img = PIL.Image.fromarray(div_img.astype(np.uint8))
    div_img.show()
    

# new_img_np = modify_image(img_np)
# new_img = PIL.Image.fromarray(new_img_np.astype(np.uint8))
# new_img.show()
# test_grad(new_img)


#u = np.array([[1, 2, 3], [4, 5, 6]])
#v = np.array([[1, 2, 3], [4, 5, 6]])
#print(test_div_grad( u, v)) 

def test_noise(img_np):
    new_img_np = add_noise(img_np, 10)

    new_img = PIL.Image.fromarray(new_img_np.astype(np.uint8))
    new_img.show()


def get_J_image_func(v, lmbd: float):
    def f(u):
        return (1/2) * np.linalg.norm(v - u)**2 + (lmbd/2) * grad_norm(u)**2
    
    def df(u):
        return u - v - lmbd * div(grad(u))

    def ddf(u):
        return np.eye(u.shape[0]) + lmbd * div(grad(u))
    
    return Function(f, df, ddf)

def test_methods(methods: tuple[METHOD_TYPE, ...], img_np: np.ndarray):
    
    # Define the parameters
    eps = 1e-6
    niter = 100
    lmbd = 0.1

    # Initialize u
    u = np.zeros_like(img_np)
    
    # Get the J function
    J = get_J_image_func(img_np, lmbd)

    # Plot the results
    
    plt.imshow(img_np)
    plt.title('Original image')
    plt.axis('off')
    plt.show()

    for method in methods:

        # Initialize u
        u = np.zeros_like(img_np)

        # Initialize list to store the values of J(u)
        J_values = []

        for i in range(niter):
        
            # Update u
            us = method(J, u, eps, 1000)
            u_min = us[-1]

            img = PIL.Image.fromarray(u_min.astype(np.uint8))
            img.save(f"figure/{METHODS_LABEL_PATH[method][1]}.png")

    
def test_plot_objective(img):
    eps = 1e-6
    niter = 100
    lmbd = 0.1

    # Initialize u
    u = np.zeros_like(img)

    # Initialize list to store the values of J(u)
    J_values = []

    # Get the J function
    J = get_J_image_func(img, lmbd)

    for i in range(niter):
        # Calculate the value of J(u)
        J_u = J.f(u)

        # Save the value of J(u)
        J_values.append(J_u)

        # Update u
        u = u - eps * J.df(u)

    # Plot the evolution of J(u)
    plt.plot(J_values)
    plt.xlabel('Iteration')
    plt.ylabel('J(u)')
    plt.title('Evolution of the objective function J(u)')
    plt.show()



def main():
    path = 'Images'
    img = PIL.Image.open(f"{path}/lena.png")
    # test_hist(img)
    # test_div(img)
    test_div_grad(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]]))
    # test_plot_objective(np.array(img))
    # test_methods((quadratic_conjuguate_gradient,), np.array(img))


def test_computePhi():
    s = np.linspace(-2, 2, 400) # On pourra changer les valeurs de s pour voir l'effet sur la fonction Phi
    # alphas = [0.01, 0.25, 0.5, 1, 1.25, 1.5, 2] # feur
    alphas = np.linspace(0.01, 2, 100)
    display_phi(s, alphas)
    plt.show()


if __name__ == '__main__':
    test_computePhi()
    # main()
  
    
    
    


