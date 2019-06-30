# python classification.py -i imagename
# example: python classification.py -i cat.png
#
# imports and some variables
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import argparse
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf


# arguments for image selection

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="name of the user")
args = vars(ap.parse_args())


# setting variables

IMG_SIZE = 50
LR = 1e-3
CATEGORIES = ["Dog", "Cat"]
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic')
train_data = np.load('train_data.npy') # loading existing train data


# tensorflow deep learning algorithm


tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


# loading model if it exists

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model loaded.')
else:
	print("Check your model directory or create new model from train.py")


# starting actual recognition progress

test_data = np.load('test_data.npy', allow_pickle=True)
img_array = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
pred_img = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
prediction = model.predict(pred_img)
dogruluk = np.amax(prediction)
if np.argmax(prediction[0]) == 1:
	label = "Dog"
else:
	label = "Cat"
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (1,25)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
cv2.putText(img_array,("{} %{:.0f}".format(label, dogruluk*100)), 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)
print("{} %{:.0f}".format(label, dogruluk*100))
cv2.imshow('Photo prediction', img_array)
cv2.waitKey(0)