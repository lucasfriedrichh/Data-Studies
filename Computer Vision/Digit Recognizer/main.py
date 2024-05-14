"""
Created on Mon May 13 21:32:40 2024

@author: Lucas Friedrich

Score: 99%
"""

import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

mnist_train = pd.read_csv("train.csv")
mnist_test = pd.read_csv("test.csv")

# See the shape os the train and test data
print(mnist_test.shape, mnist_train.shape)
mnist_train.head()
mnist_train.describe()
    
# All the data cleaning and normalization 
mnist_train.isna().any().any()
# since the final result says False which means it has no missing values so data is clear

mnist_train_data = mnist_train.loc[:, 'pixel10':] #All rows and all columns starting in pixel10 to the end
mnist_train_label = mnist_train.loc[:, 'label']

# Notmailzing the images array to be in the range of 0-1 by dividing them by the max possible value. 
# Here is it 255 as we have 255 value range for pixels of an image. 
mnist_train_data = mnist_train_data / 255.0
mnist_test = mnist_test/255.0

#Creating the plots
digit_array = mnist_train.loc[3, "pixel0":]
arr = np.array(digit_array)

#.reshape(a, (28,28))
image_array = np.reshape(arr, (28,28))

digit_img = plt.imshow(image_array, cmap=plt.cm.binary)
plt.colorbar(digit_img)
print("Image level: {}".format(mnist_train.loc[3, 'label']))