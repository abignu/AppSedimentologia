#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request 
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential  
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils.np_utils import to_categorical  
import matplotlib.pyplot as plt  
import math  
import cv2  

#scientific computing library for saving, reading, and resizing images
#from scipy.misc import imsave, imread, imresize
#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re
import base64
#system level operations (like loading files)
import sys 
#for reading operating system data
import os, glob

from keras.applications.vgg16 import preprocess_input
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from load import * 
#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
global model, graph
from PIL import Image
import cv2

from werkzeug.utils import secure_filename
#initialize these variables
model, graph = init()

#decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    #print(imgstr)
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgstr))

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    '''
    args: a path to the image
    returns: a 4D tensor
    '''
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)

    x = cv2.resize(x,(224, 224))

    # important! otherwise the predictions will be '0'  
    x = x / 255 
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)    

def rock_prediction(img):
    rocks = ['Fosil', 'No-Fosil']
    bottleneck_features = applications.VGG16(include_top=False, weights='imagenet').predict(img)
    preds = model.predict(bottleneck_features)
    print(preds)
    return rocks[np.argmax(preds)]

@app.route('/')
def index():
    #initModel()
    #render out pre-built HTML file right on the index page
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    #whenever the predict method is called, we're going
    #to input the user drawn character as an image into the model
    #perform inference, and return the classification
    #get the raw data format of the image
    # Get the image file from the post request
    file = request.files['file']
        
        # Save the file with proper image path
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, secure_filename(file.filename))
    file.save(file_path)
    image = path_to_tensor(file_path)
    #image = preprocess_input(image)
    
    #if detecting(image):
    with graph.as_default():
        prediction = rock_prediction(image)
        result = str(prediction)
        print(result)
    
        # if face_detecting(image):
        #     result = 'This photo looks like a/an {}'.format(prediction)
    
        
        #result = 'So sorry: This image is not recognizable!'
        
        result = "It's a " + result

    # else: 
    #     result = "This doesn't seem to be a dog --- Sorry!"
        
        #eliminamos image
        for fname in glob.glob('*.png'):
            os.remove(fname) 

        return result
    

if __name__ == "__main__":
    #decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    #run the app locally on the givn port
    app.run(host='0.0.0.0', port=port)
    #optional if we want to run in debugging mode
    #app.run(debug=True)
