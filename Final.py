# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:25:38 2020

@author: deepak
"""
####  Importing The Libraries

from matplotlib import pyplot as plt
import numpy as np

### Import tensor flow libraries

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Sequential


import os
import cv2

from tqdm import tqdm

from keras.preprocessing.image import img_to_array

np.random.seed(42)

SIZE = 28
#### Import noisy data set

noisy_data =[]
path1 = "images/Nimages/"
files = os.listdir(path1)
for i in tqdm(files):
    img = cv2.imread(path1+'/'+i, 0)
    img = cv2.resize(img, (SIZE, SIZE))
    noisy_data.append(img_to_array(img))

### Import Clean data set

clean_data =[]
path2 = "images/Color_Images/"
files = os.listdir(path2)
for i in tqdm(files):
    img = cv2.imread(path2+'/'+i, 0)
    img = cv2.resize(img, (SIZE,SIZE))
    clean_data.append(img_to_array(img))

### split in training and test sets

noisy_train = np.reshape(noisy_data, (len(noisy_data)), SIZE, SIZE, 1)
noisy_train = noisy_train.astype('float32')/255.

clean_train = np.reshape(clean_data, (len(clean_data)), SIZE, SIZE, 1)
clean_train = clean_train.astype('float32')/255.


###Display image with noise
plt.figure(figsize = (10,2))
for i in range(1,4):
    ax = plt.subplot(1,4, i)
    plt.imshow(noisy_train[i].reshape(SIZE, SIZE), cmap = 'binary')
plt.show()


###Display clean images
plt.figure(figsize = (10,2))
for i in range(1,4):
    ax = plt.subplot(1,4, i)
    plt.imshow(clean_train[i].reshape(SIZE, SIZE), cmap = 'binary')
plt.show()



#### Create Auto-Encoder model of Nural Networks

### Intilization

model = Sequential()

### Encoding Model

model.add(Conv2D(32, (3, 3), activation = 'relu', input_size = (SIZE, SIZE, 1)))
model.add(MaxPooling2D((2,2), padding = 'same'))

model.add(Conv2D(8, (3,3), activation ='relu', padding = 'same'))
model.add(MaxPooling2D((2,2), padding = 'same'))

model.add(Conv2D(8, (3,3), activation ='relu', padding = 'same'))
model.add(MaxPooling2D((2,2), padding = 'same'))



### Decoding Model

model.add(Conv2D(8, (3,3), activation ='relu', padding = 'same'))
model.add(UpSampling2D((2,2)))

model.add(Conv2D(8, (3,3), activation ='relu', padding = 'same'))
model.add(UpSampling2D((2,2)))

model.add(Conv2D(32, (3,3), activation ='relu', padding = 'same'))
model.add(UpSampling2D((2,2)))

### Output Layer

model.add(Conv2D(1, (3,3), activation = 'relu', padding = 'same'))

#### Compile the model

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrices = ['accuracy'])

model.summary()

### Spliting The training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(noisy_data, clean_data, test_size = 0.2, random_state = 0)


model.fit(X_train, y_train, epochs = 10, batch_size = 8, shuffle = 'True',
          verbose = 1, validation_split = 0.1)

print("Test_Accuracy : {:, 2f}%".format(model.evaluate(np.array(X_test), np.array(y_test))[1]*100))

model.save('denoiseing_autoencoder.model')

no_noise_img = model.predict(X_test)

plt.imshow(no_noise_img[i].reshape(SIZE, SIZE), cmap = 'gray')

### PSNR

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

for i in range(len(noisy_data)):
    PSNR = psnr(noisy_data[i] - no_noise_img[i])
    print(PSNR)
