from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
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

IMG_SIZE = 50
LR = 1e-3
CATEGORIES = ["Dog", "Cat"]
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic')
train_data = np.load('train_data.npy') # loading existing train data

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

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model loaded.')
else:
	print("Check your model directory or create new model from train.py")

test_data = np.load('test_data.npy', allow_pickle=True)

def select_image():
	global panelA, panelB

	path = filedialog.askopenfilename()

	if len(path) > 0:
		image = cv2.imread(path, cv2.COLOR_BGR2RGB)
		img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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
		fontColor              = (0,255,255)
		lineType               = 2
		cv2.putText(img_array,("{} %{:.0f}".format(label, dogruluk*100)), 
		    bottomLeftCornerOfText, 
		    font, 
		    fontScale,
		    fontColor,
		    lineType)

		image = Image.fromarray(image)
		edged = Image.fromarray(img_array)

		image = ImageTk.PhotoImage(image)
		edged = ImageTk.PhotoImage(edged)

		if panelA is None or panelB is None:
			panelA = Label(image=image)
			panelA.image = image
			panelA.pack(side="left", padx=10, pady=10)

			panelB = Label(image=edged)
			panelB.image = edged
			panelB.pack(side="right", padx=10, pady=10)

		else:
			# update the pannels
			panelA.configure(image=image)
			panelB.configure(image=edged)
			panelA.image = image
			panelB.image = edged

root = Tk()
panelA = None
panelB = None

btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

root.mainloop()
