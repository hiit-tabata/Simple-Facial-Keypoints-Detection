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
from keras.optimizers import SGD, Adadelta


TRAINING_SET = './training.csv'
TEST_SET = './test.csv'



predicted_names = ["left_eye_center_x","left_eye_center_y","right_eye_center_x","right_eye_center_y","left_eye_inner_corner_x","left_eye_inner_corner_y","left_eye_outer_corner_x","left_eye_outer_corner_y","right_eye_inner_corner_x","right_eye_inner_corner_y","right_eye_outer_corner_x","right_eye_outer_corner_y","left_eyebrow_inner_end_x","left_eyebrow_inner_end_y","left_eyebrow_outer_end_x","left_eyebrow_outer_end_y","right_eyebrow_inner_end_x","right_eyebrow_inner_end_y","right_eyebrow_outer_end_x","right_eyebrow_outer_end_y","nose_tip_x","nose_tip_y","mouth_left_corner_x","mouth_left_corner_y","mouth_right_corner_x","mouth_right_corner_y","mouth_center_top_lip_x","mouth_center_top_lip_y","mouth_center_bottom_lip_x","mouth_center_bottom_lip_y"]


def loadTrainingset(filePath):
    df = pd.read_csv(os.path.expanduser(filePath))
    df = df.dropna()

    imgs = df["Image"].apply(lambda im: np.fromstring(im, sep=' '))
    imgs = np.vstack(imgs.values)/255 - 0.5
    imgs = imgs.astype(np.float32)

    labels = df[df.columns[:-1]].values
    labels = (labels-48)/48
    imgs, labels = sklearn.utils.shuffle(imgs, labels, random_state=42)
    labels = labels.astype(np.float32)

    imgs = imgs.reshape(-1,96,96,1)
    return imgs, labels


def loadTest(path):
    df = pd.read_csv(os.path.expanduser(path))
    df = df.dropna()

    imgs = df["Image"].apply(lambda im: np.fromstring(im, sep=' '))
    imgs = np.vstack(imgs.values)/255- 0.5
    imgs = imgs.astype(np.float32)
    imgs = imgs.reshape(-1,96,96,1)
    return imgs




imgs, labels = loadTrainingset(TRAINING_SET)

plt.imshow(imgs[2].reshape(96,96), cmap=plt.get_cmap('gray'))

plt.tight_layout()
plt.show()

print(imgs.shape)
print(labels.shape)

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(96, 96,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, 2, 2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(128, 2, 2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(30))

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
hist = model.fit(imgs, labels, nb_epoch=100, validation_split=0.2)

f=plt.figure()
plt.plot(hist.history['loss'], linewidth=3, label='train')
plt.plot(hist.history['val_loss'], linewidth=3, label='valid')
plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.yscale('log')
plt.show()
f.savefig('./loss.png')



def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2], y[1::2], marker='x', s=10)

#predict test images
X_test = loadTest(TEST_SET)
y_test = model.predict(X_test)
y_test=y_test*48+48
fig = plt.figure(figsize=(6,6))
for i in range(16):
    axis = fig.add_subplot(4,4,i+1,xticks=[],yticks=[])
    plot_sample(X_test[i], y_test[i], axis)
    fig.show()
    pass
    

def getPredicted(predictedResult, imgId, position):
    '''
        imgId should = matrix index +1, shince the img id count from 1
    '''
    return predictedResult[imgId-1][predicted_names.index(position)]


idLoopUp = pd.read_csv(os.path.expanduser('IdLookupTable.csv'))
idLoopUp = idLoopUp.as_matrix()
for i in range(len(idLoopUp)):
    idLoopUp[i][3] = getPredicted(y_test, idLoopUp[i][1],idLoopUp[i][2])

result = idLoopUp[:,[0,3]]
result = result.astype(int)
np.savetxt('keypoints_pred.csv', result, delimiter=',', header="RowId,Location", fmt='%2i')