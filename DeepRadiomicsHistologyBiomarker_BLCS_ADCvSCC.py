### This script allows for prediction of histology using a pre-built biomaker based on ADC vs SCC
### using this we then tell apart 3 different histologies

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from tempfile import TemporaryFile
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, SelectFromModel
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.models import Model


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

# functions

def non_shuffling_train_test_split(X, y, test_size=0.2):
    i = int((1 - test_size) * X.shape[0]) + 1
    X_train, X_test = np.split(X, [i])
    y_train, y_test = np.split(y, [i])
    return X_train, X_test, y_train, y_test

def AUC(test_labels,test_prediction,nb):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb):
        # ( actual labels, predicted probabilities )
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], test_prediction[:, i] ) # flip here
        roc_auc[i] = auc(fpr[i], tpr[i])

    return [ round(roc_auc[x],3) for x in range(nb) ] 
#test_size = .3630 
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
Outcomes_file = '/home/chintan/Desktop/AAPM/ModelsandData/Data/AdenoSquamALL.csv' #define the outcomes file, sorted  according to PID Stage1and2CompositeTest
path1 = '/home/chintan/Desktop/AAPM/ModelsandData/Data/Stage1and2ALL'    #path of folder of images    
path2 = '/home/chintan/Desktop/AAPM/ModelsandData/Data/TestImageCrops' #path of folder to save images    
Outcomes = pd.read_csv(Outcomes_file) #outcomes file is sorted so as to match the image index
Outcome_of_interest = pd.Series.as_matrix(Outcomes.loc[:,'Adeno']) #pick the column with the labels of interest
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
#modelFile = (os.path.join(predDir,'VGG_AdenovSCC_FINAL.h5')) #VGG biomarker:  (VGG_AdenovSCC_FINAL.h5)

model = load_model(modelFile)

### Extract features from layer M ###

layer_index = 20 # 
func1 = K.function([ model.layers[0].input , K.learning_phase()  ], [ model.layers[layer_index].output ] )

#Feat = np.empty([1,1,nb_classes]) #when layer_index =24
Feat = np.empty([1,1,4096]) # when layer_index =20
#Feat = np.empty([1,1,512]) # when layer_index =19
for i in xrange(X.shape[0]):
	input_image=X[i,:,:,:].reshape(1,img_rows,img_cols,3)
	input_image_aslist= [input_image,0]
	func1out = func1(input_image_aslist)
	features = np.asarray(func1out)
	#print features
	Feat = np.concatenate((Feat, features), axis = 1)


Feat = squeeze(Feat)
Features = Feat[1:Feat.shape[0],:]

### Dimensionality reduction with PCA and LASSO ####
num_best_features = 60 #set to 60 to capture 90% of variance 29 for 75% variance, 10 for 50% of variance

random.seed(10) # set random starting point 
pca = PCA(n_components = num_best_features)
pcam = pca.fit(Features,y)
F = pcam.transform(Features)

plt.figure(1)
plt.plot(np.cumsum(pcam.explained_variance_ratio_))
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance');
plt.show()

## LASSO-Cox selection of features with strong association with class
llas = linear_model.Lasso(alpha= 0.01).fit(F, y)
feature_model = SelectFromModel(llas, prefit=True)
F = feature_model.transform(F)

## Feature normalization
from sklearn import preprocessing 
std_scale = preprocessing.StandardScaler().fit(F)
F = std_scale.transform(F)

### Machine learning classifiers ###
test_size = 0.23
F_train, F_test, y_train, y_test = non_shuffling_train_test_split(F, y, test_size=test_size) #use non random data splitting 

## Run best features on the Machine learning classifier model ##
x_train = F_train#X_new_train
y_train = y_train

x_test = F_test #X_new_test
y_test = y_test

#Define model: uncomment the model of interest below

#ols = linear_model.Lasso(alpha =0.1) #LASSO
#ols = RandomForestClassifier() # Random Forrest
#ols = svm.LinearSVC()  #Support vector machine, also try SVC! 
#ols = svm.SVC(kernel="linear", C=0.01)
#ols = svm.SVC()
ols = KNeighborsClassifier(n_neighbors=5) # set this to odd number 

ml_model = ols.fit(x_train, y_train) #define machine learning model

y_pred = ml_model.predict(x_test)

###
Y_pred = np_utils.to_categorical(y_pred, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
ROC = AUC(Y_test,Y_pred, nb_classes)
print ('AUC =', ROC[1])

import scipy.stats as stat
a = Y_pred[:, 0]
b = Y_test[:, 0]
groups = [a[b == i] for i in xrange(2)]
rs = stat.ranksums(groups[0], groups[1])[1]
print('p = ',rs)

from sklearn.metrics import confusion_matrix
from fractions import Fraction

cm1 = confusion_matrix(y_test, np.round(Y_pred[:,1]))

total1=float(sum(sum(cm1)))

accuracy1= (cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = float(cm1[0,0])/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = float(cm1[1,1])/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)


## plot feature histogram
plt.figure(2)
plt.style.use('seaborn-deep')
plt.hist([F_train[:,7], F_test[:,7]], label=['Train', 'Test'])
plt.legend(loc='upper right')
plt.ylabel('Frequency')
plt.xlabel('Abstract feature value')
plt.show()
         
