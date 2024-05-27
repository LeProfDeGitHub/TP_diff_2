import PIL.Image
import numpy as np 
import PIL


path = 'Images/Images'

img = PIL.Image.open(f"{path}/lena.png")

img.show()

img = np.array(img)
print(img.shape)