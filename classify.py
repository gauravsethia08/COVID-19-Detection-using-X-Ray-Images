#Importing Libraries
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import time

#Constructuing the Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True, help = "path to model")
ap.add_argument("-l", "--labelbin", required = True, help = "path to label binarizer file")
ap.add_argument("-i", "--image", required = True, help = "path to input image")
args = vars(ap.parse_args())

#Loading the image
image = cv2.imread(args["image"])
output = image.copy()

#Preprocessing the image
image = cv2.resize(image, (96, 96))
image = image.astype("float")/255.0
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)

#loading the trained convolutional neural network and the label binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

start = time.time()
#Classifying image
print("[INFO] classifying image...")
pred = model.predict(image)
proba = pred[0]
stop = time.time()

	
idx = np.argmax(proba)
label = lb.classes_[idx]	
		
os.system("clear")
print("Leaf is -  {}".format(label+1))
print("Accuracy is {:}".format(100*proba[idx])) 
print("Time is {}".format(stop-start))
