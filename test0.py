### import the libraries

"""import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


from tensorflow.keras .datasets import mnist
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Sequential

(X_train, _),(X_test, _) = mnist.load_data()

X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.

X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))

###Add artifical noise 

###RAndom noise from normal distribution method with mean at 0 and std dev of 1

noise_factor = 0.5
X_train_noisy = X_train + np.random.normal(loc = 0.0, scale = 1.0, size = X_train.shape)
X_test_noisy = X_test + np.random.normal(loc = 0.0, scale = 1.0, size = X_test.shape)


X_train_noisy = np.clip(X_train_noisy, 0.0, 1.0, out = None)
X_test_noisy = np.clip(X_test_noisy, 0.0, 1.0, out = None)


plt.figure(figsize =(20,2))
for i in range(1,10):
    ax = plt.subplot(1, 10, i)
    plt.imshow(X_test_noisy[i].reshape(28, 28), cmap = 'binary')
plt.show()
"""
"""path1 = "Noisy_data/*"
for file in glob.glob(path1):
    imageA = cv2.imread(file)

path2 = "clean_data/*"
for file in glob.glob(path2):
    imageB = cv2.imread(file)
    cv2.imshow("Clean", imageB)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 """   

import cv2
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from matplotlib import pyplot as plt
import glob

imageA = cv2.imread('Noisy_data/i2.png')
imageB = cv2.imread('clean_data/ii2.png')

s = ssim(imageA, imageB, multichannel = True)
t = ssim(imageA, imageA, multichannel = True)
p = psnr(imageB, imageA)

