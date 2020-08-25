from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img_arr= np.load('.npy)

im= Image.fromarray(img_arr[0][1])

im.show()