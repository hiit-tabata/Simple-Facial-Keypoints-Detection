# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from numpy import genfromtxt


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD


TRAINING_SET = './training.csv'
TEST_SET = './test.csv'


def loadTrainingset(filePath):
    df = pd.read_csv(os.path.expanduser(filePath))
    df = df.dropna()
    
    imgs = df["Image"].apply(lambda im: np.fromstring(im, sep=' '))
    imgs = np.vstack(imgs.values)/255
    imgs = imgs.astype(np.float32)
    
    labels = df[df.columns[:-1]].values
    labels = (labels-48)/48
    imgs, labels = sklearn.utils.shuffle(imgs, labels, random_state=42)
    labels = labels.astype(np.float32)
    
    imgs = imgs.reshape(-1,96,96,1)
    return imgs, labels 


imgs, labels = loadTrainingset(TRAINING_SET)

plt.imshow(imgs[2].reshape(96,96), cmap=plt.get_cmap('gray'))

plt.tight_layout()
plt.show()

