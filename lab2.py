import numpy as np
import pandas as pd
import keras

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from matplotlib.pyplot import *
import matplotlib.pyplot as plt

import os, re
from pprint import pprint

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils


test_df = pd.read_csv('optdigits.tes', header=None)
train_df = pd.read_csv('optdigits.tra', header=None)

Xtrain = np.array(train_df.drop([64], axis=1))
Xtest = np.array(test_df.drop([64], axis=1))

ytrain = np.array(train_df[64])
ytest = np.array(test_df[64])

def get_mean(X_std):
    X_mean = np.mean(X_std, axis=0)
    return X_mean


def get_standardized(X):
    X_std = StandardScaler().fit_transform(X)
    return X_std


def get_cov(X_std):
    X_cov = np.cov(X_std.T)
    return X_cov

# X_train == X_std

Xtrain = get_standardized(Xtrain)
X_train_mean = get_mean(Xtrain)
X_train_cov = get_cov(Xtrain)

Xtest = get_standardized(Xtest)
X_test_mean = get_mean(Xtest)
X_test_cov = get_cov(Xtest)

ytrain = np_utils.to_categorical(ytrain, 10)
ytest = np_utils.to_categorical(ytest, 10)

print(Xtrain[0].shape)

# for i in range(9):
#     plt.subplot(331+i)
#     plt.imshow(X_train.reshape(-1,1,8,8)[i][0], cmap=cm.bone)
#     gca().grid(False)
# plt.show()

# 3 layers with softmax
# print(train_df)
# print(test_df)

input_shape = Xtrain[0].shape

model = Sequential([
  Dense(64, activation='tanh', input_shape=input_shape),
  Dense(64, activation='tanh'),
  Dense(10, activation='softmax'),
])
#
#
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)
# print(y_train.shape)
#
model.fit(
  Xtrain,
  ytrain,
  epochs=20,
  validation_split=0.2,
  verbose=2,
)















