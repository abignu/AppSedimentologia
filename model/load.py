
import keras.models
from keras.models import model_from_json
#from scipy.misc import imread, imresize,imshow
import tensorflow as tf

import numpy as np  
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.models import Sequential  
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils.np_utils import to_categorical  
import matplotlib.pyplot as plt  
import math  
import cv2  

def init():
	top_model_weights_path = 'bottleneck_fc_model_weights.h5'

	num_classes = 2

	# load the bottleneck features saved earlier  
	  
	#codigo SIRAJ
	'''
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load woeights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	'''
	model = Sequential()  
	model.add(Flatten(input_shape=(7, 7, 512)))  
	model.add(Dense(256, activation='relu'))  #FC layer
	model.add(Dropout(0.5))  
	model.add(Dense(num_classes, activation='sigmoid'))  

	model.compile(optimizer='rmsprop',  
			  loss='binary_crossentropy', metrics=['accuracy'])

	model.load_weights(top_model_weights_path)

	graph = tf.get_default_graph()

	return model, graph