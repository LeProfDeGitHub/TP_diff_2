import PIL.Image
import numpy as np 
import PIL
from image_methods import (modify_image,
                           grad,
                           retourner_image, 
                           test_div_grad,
                           add_noise,)



path = 'Images/Images'
img = PIL.Image.open(f"{path}/lena.png")


img_np = np.array(img)
print(img_np.shape)
hist5 = np.histogram(img_np, bins=5, range=(0, 5))
print(hist5)
his256 = np.histogram(img_np, bins=256, range=(0, 256))    
print(256)


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
    gx_img.show()
    gy_img.show()

# new_img_np = modify_image(img_np)
# new_img = PIL.Image.fromarray(new_img_np.astype(np.uint8))
# new_img.show()
# test_grad(new_img)


#u = np.array([[1, 2, 3], [4, 5, 6]])
#v = np.array([[1, 2, 3], [4, 5, 6]])
#print(test_div_grad( u, v)) 


new_img_np = add_noise(img_np, 10)

new_img = PIL.Image.fromarray(new_img_np.astype(np.uint8))
new_img.show()




