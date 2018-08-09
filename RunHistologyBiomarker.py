### This script allows for prediction of histology using a pre-built biomaker based on ADC vs SCC
### using this we then tell apart 3 different histologies

#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from keras import backend as K #this just makes stuff work
#K.set_image_dim_ordering('th')
from keras.applications.vgg16 import preprocess_input

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image

# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from tempfile import TemporaryFile
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

#from other libraries
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chisquare
import scipy.stats as stats

### DATA ###
nb_classes = 2
#input image dimensions
#input image dimensions
OG_size = 150 #original image size 
Center = OG_size/2
img_rows_OG, img_cols_OG = 50, 50
x1, y1, x2, y2 = Center-img_rows_OG/2,Center-img_rows_OG/2,Center+img_cols_OG/2,Center+img_cols_OG/2 

# number of channels
img_channels = 3

img_rows, img_cols = 50, 50 # new input image dimensions! Different from image dimensions OG 
# number of channels
img_channels = 1
# data
Outcomes_file = '/home/chintan/Desktop/AAPM/ModelsandData/Data/FinalModels/Stage1and2CompositeTest.csv' #define the outcomes file, sorted  according to PID Stage1and2CompositeTest
path1 = '/home/chintan/Desktop/AAPM/ModelsandData/Data/Stage1and2ALL'    #path of folder of images    
path2 = '/home/chintan/Desktop/AAPM/ModelsandData/Data/TestImageCrops' #path of folder to save images    
Outcomes = pd.read_csv(Outcomes_file) #outcomes file is sorted so as to match the image index
Outcome_of_interest = pd.Series.as_matrix(Outcomes.loc[:,'AdenovNon']) #pick the column with the labels of interest
PID = pd.Series.as_matrix(Outcomes.loc[:,'PID'])

listing = os.listdir(path1) 
num_samples=size(listing)
print num_samples


for file in listing:
    im = Image.open(path1 + '/' + file) 
    imx = im.crop((x1,y1,x2,y2))  #crop the images to x1-x2 x y1-y2 centered on tumor
    img = imx.resize((img_rows,img_cols), Image.NEAREST)
    gray = img.convert('RGB')  #gray is a misnormer since we are converting to RGB        
    gray.save(path2 +'/' +  file, "PNG")

filename = []
for i in range(len(PID)):
	filenames = str(PID[i])+'_Axial.png'
	#filenames = dataset[i]+'_'+str(PID[i])+ '.png' #Use this line if not BLCS
	filename = np.append(filename, [filenames])
	#print filename

imlist = filename

im1 = array(Image.open( path2+ '/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open(path2+ '/' + im2)).flatten()
              for im2 in imlist],'f')             
              
#define labels as the outcomes

label = Outcome_of_interest

data,Label = immatrix,label
train_data = [data,Label]

print (train_data[0].shape)
print (train_data[1].shape)

(X, y) = (train_data[0],train_data[1])
X = X.reshape(X.shape[0], img_rows, img_cols,3)
X= X.astype('float32')

X /= 255
#X = preprocess_input(X)


print('X shape:', X.shape)
print(X.shape[0], 'test samples')


### MODEL###

## load pretrained model ##


predDir = '/home/chintan/Desktop/AAPM/ModelsandData/Data/FinalModels/'
modelFile = (os.path.join(predDir,'AdenovSCC_Final_THEMODEL.h5')) #VGG biomarker


model = load_model(modelFile)

### Extract features from layer M ###

layer_index = 24 # 180 is the layer number corresponding to softmax layer
func1 = K.function([ model.layers[0].input , K.learning_phase()  ], [ model.layers[layer_index].output ] )

Feat = np.empty([1,1,nb_classes]) #when layer_index =19
#Feat = np.empty([1,1,4096]) # when layer_index =20
for i in xrange(X.shape[0]):
	input_image=X[i,:,:,:].reshape(1,img_rows,img_cols,3)
	input_image_aslist= [input_image,0]
	func1out = func1(input_image_aslist)
	features = np.asarray(func1out)
	#print features
	Feat = np.concatenate((Feat, features), axis = 1)


Feat = squeeze(Feat)
Features = Feat[1:Feat.shape[0],:]

### 
F = Features

from sklearn.metrics import roc_curve, auc

def AUC(test_labels,test_prediction,nb):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb):
        # ( actual labels, predicted probabilities )
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], test_prediction[:, i] ) # flip here
        roc_auc[i] = auc(fpr[i], tpr[i])

    return [ round(roc_auc[x],3) for x in range(nb) ] 

Y_pred = F
Y = np_utils.to_categorical(label, nb_classes)
ROC = AUC(Y,Y_pred, nb_classes)
print ('AUC =', ROC[1])

import scipy.stats as stat
a = Y_pred[:, 0]
b = Y[:, 0]
groups = [a[b == i] for i in xrange(2)]
rs = stat.ranksums(groups[0], groups[1])[1]
print('p = ',rs)

score = model.evaluate(X, Y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#find indeces where Y is greater than a certain value

#indeces = [i for i, v in enumerate(a>=0.5) if v]


#taf = 1-sum(b[a<=median(a)])/b[a<=median(a)].shape[0]
#print('Taf Satisfaction index', taf)


#3 groups histology. Determine significance of middle predictions for "other" category

Hist3 = pd.Series.as_matrix(Outcomes.loc[:,'HIST3TIER'])
Histology = np_utils.to_categorical(Hist3, 3)

#Define histologies
SCC = Histology[:,2]
ADC = Histology[:,1]
Other = Histology[:,0]

c = Y_pred[:, 1] ## c = biomaker predictions. bigger numbers are predictive of adenocarcinoma, smaller ones are predictive of SCC, other is in the middle

from scipy.stats import iqr
# seperate predictions into 3 quartiles
U_iqr = np.percentile(c,42.2) # upper limit of inter thirdile range
L_iqr = np.percentile(c,21.7) # lower limit of inter thirdile range



SCC_preds = sum(SCC[c<=L_iqr])/SCC[c<=L_iqr].shape[0]
ADC_preds = sum(ADC[c>=U_iqr])/ADC[c>=U_iqr].shape[0]
print('SCC prediction concordance', SCC_preds)
print('ADC prediction concordance', ADC_preds)


Other2= Other[c<=U_iqr]
c2 = c[c<=U_iqr]

Other3 = Other2[c2>=L_iqr]

Other_preds = sum(Other3)/Other3.shape[0]
print('Other prediction concordance', Other_preds)

print("#Other, #Sample, Original#Other, Original#Sample ", sum(Other3), Other3.shape[0], sum(Other), Other.shape[0])

#run numbers returned above with: https://www.medcalc.org/calc/comparison_of_proportions.php

