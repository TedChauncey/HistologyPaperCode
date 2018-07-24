### this script is a work in progress! It does not work 
########################################

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from sklearn.metrics import roc_curve, auc

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

from PIL import Image
from numpy import *
from keras import backend as K


from fractions import Fraction
import scipy.ndimage

## Define functions ##
def AUC(test_labels,test_prediction,nb):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb):
        # ( actual labels, predicted probabilities )
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], test_prediction[:, i] ) # flip here
        roc_auc[i] = auc(fpr[i], tpr[i])

    return [ round(roc_auc[x],3) for x in range(nb) ] 
    
def AUCalt( test_labels , test_prediction):
    # convert to non-categorial
    test_prediction = np.array( [ x[1] for x in test_prediction   ])
    test_labels = np.array( [ 0 if x[0]==1 else 1 for x in test_labels   ])
    # get rates
    fpr, tpr, thresholds = roc_curve(test_labels, test_prediction, pos_label=1)
    # get auc
    myAuc = plt.plot(fpr, tpr)
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.legend(['ROC','FPR'],loc=4)
    plt.show()
    #myAuc = auc(fpr, tpr)
    return myAuc
 
img_rows, img_cols = 200, 200 # new input image dimensions! Different from image dimensions OG 
#MODEL Parameters ##
#batch_size to train
batch_size = 64
# number of output classes
nb_classes = 2
# number of epochs to train
nb_epoch = 100

#### DEFINE MODEL########
model_base = ResNet50(weights='imagenet', include_top=False, input_shape = (img_rows,img_cols,3) )
model_base.layers.pop()
model_base.outputs = [model_base.layers[-1].output]
model_base.layers[-1].outbound_nodes =[]

model_top = model_base.output
model_top = Flatten()(model_top)
model_top =Dense(2048, activation = 'relu')(model_top)
model_top = Dropout(0.5)(model_top)
model_top =Dense(2048, activation = 'relu')(model_top)
model_top = Dropout(0.5)(model_top)
model_top =Dense(nb_classes, activation = 'softmax')(model_top)

model = Model(input = model_base.input, output=model_top)
for layer in model.layers[:155]:   #175 is all non fully connected layers
    layer.trainable = False

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics = ['accuracy'])


### DEFINE DATA ###

#input image dimensions
OG_size = 150 #original image size 
Center = OG_size/2
img_rows_OG, img_cols_OG = 50, 50
x1, y1, x2, y2 = Center-img_rows_OG/2,Center-img_rows_OG/2,Center+img_cols_OG/2,Center+img_cols_OG/2 

# number of channels
img_channels = 3

## DATA SOURCES ##

Outcomes_file = '/home/ahmed/Taf/AdenoSCCData/Stage1and2Train.csv' #define the outcomes file, sorted according to PID
path1 = '/home/ahmed/Taf/AdenoSCCData/Stage1and2ALL'    #path of folder of images    
path2 = '/home/ahmed/Taf/AdenoSCCData/TestImageCrops' #path of folder to save images    
Outcomes = pd.read_csv(Outcomes_file) #outcomes file is sorted so as to match the image index
Outcome_of_interest = pd.Series.as_matrix(Outcomes.loc[:,'Adeno']) #pick the column with the labels of interest
PID = pd.Series.as_matrix(Outcomes.loc[:,'PID'])
#dataset = pd.Series.as_matrix(Outcomes.loc[:,'dataset']) #comment this out when its BLCS

listing = os.listdir(path1) 
num_samples=size(listing)
print num_samples

for file in listing:
    im = Image.open(path1 + '/' + file) 
    imx = im.crop((x1,y1,x2,y2))  #crop the images to x1-x2 x y1-y2 centered on tumor
    img = imx.resize((img_rows,img_cols), Image.NEAREST)
    gray = img.convert('RGB')  #gray is a misnormer since we are converting to RGB        
    gray.save(path2 +'/' +  file, "PNG")

##change below lines depending on dataset being used ..

filename = []
for i in range(len(PID)):
	filenames = str(PID[i])+'_Axial.png' # use this line if BLCS
	#filenames = dataset[i]+'_'+str(PID[i])+ '.png' #Use this line if not BLCS
	filename = np.append(filename, [filenames])
imlist = filename

im1 = array(Image.open( path2+ '/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

#create matrix to store all flattened images
immatrix = array([array(Image.open(path2+ '/' + im2)).flatten()
              for im2 in imlist],'f')             
              
#define labels as the outcomes#
label = Outcome_of_interest
data,Label = shuffle(immatrix,label, random_state = 2) #see how shuffle works
train_data = [data,Label]

from keras import backend as K #this just makes stuff work
K.set_image_dim_ordering('th')


(X, y) = (train_data[0],train_data[1])

# split X and y into training and testing sets
test_size = .20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=4)


X_train = X_train.reshape(X_train.shape[0],  img_rows, img_cols,3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#X_train /= 255
#X_test /= 255

X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test) 

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

### TRAIN THE MODEL ####

hist = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    verbose=1,
                    validation_data=(X_test, Y_test))     
    
### MODEL ASSESSMENT ###

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Neural Net Test accuracy:', score[1])

##Predictions - from test set ####
pred_results = []
for i in xrange(X_test.shape[0]):
    predictions = model.predict(np.expand_dims(X_test[i], axis = 0))
    pred_results = np.append(pred_results,[predictions])

Y_pred = pred_results.reshape(X_test.shape[0], nb_classes)
#print(Y_pred)

ROC2 = AUC(Y_test, Y_pred, nb_classes)

print ("Neural Net AUC:",ROC2[1])

#save model 
#fname = "RESNET_ScreeningModel_run6.h5"
#model.save(fname)

#determine pvalue of AUC
import scipy.stats as stat
a = Y_pred[:, 0]
b = Y_test[:, 0]
groups = [a[b == i] for i in xrange(2)]
pvalue = stat.ranksums(groups[0], groups[1])[1]

print ('pvalue:', pvalue)
####

X= X.reshape(X.shape[0],  img_rows, img_cols,3)
layer_index = 175 # 24 for prediction models, 26 for the newer deeper models. This should be 21#Set to 19 for 512-D vector, 20 for 4096-D, 24 for 24 normally
func1 = K.function([ model.layers[0].input , K.learning_phase()  ], [ model.layers[layer_index].output ] )

Feat = np.empty([1,1,2048]) #when layer_index =19
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

from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import linear_model

num_best_features = 100  # number of components after dim reduction

random.seed(10) # set random starting point . 3 and 10 is best
pca = PCA(n_components = num_best_features)
pcam = pca.fit(Features,y)
F = pcam.transform(Features)

#normalize features
from sklearn import preprocessing
F = preprocessing.scale(F)

F_train, F_test, y_train, y_test = train_test_split(F, y, test_size=test_size, random_state=4)


## Run best features on the Machine learning classifier model ##
x_train = F_train #X_new_train
y_train = y_train

x_test = F_test #X_new_test
y_test = y_test

#Define model: uncomment the model of interest below

#ols = linear_model.Lasso(alpha =0.001) #LASSO
#ols = RandomForestClassifier() # Random Forrest
#ols = svm.LinearSVR()  #Support vector machine, also try SVC! 
ols = svm.SVC()  #Support vector machine, also try SVC! 
#ols = linear_model.LinearRegression() #MVR

ml_model = ols.fit(x_train, y_train) #define machine learning model

y_pred = ml_model.predict(x_test)
y_pred[y_pred<0] = 0 #make sure predictions are never negative (i.e. assign these to 0)
y_pred[y_pred>1] =1 #make sure predictions are never greater than 1. I.e assign these to 1. 
#print(y_pred[0:8])

y_test2 = np_utils.to_categorical(y_test, nb_classes)
y_pred2 = np.row_stack([1-abs(y_pred), y_pred]).T
y_pred2 = abs(y_pred2)

acc = accuracy_score(np.asarray(y_test), np.round(y_pred))
#acc = sum(y_pred == y_test)/len(y_test)
print('Machine Learning Accuracy:', acc)

ROC2 = AUC(y_test2, y_pred2, nb_classes)
print('Machine learning AUC:', ROC2[1]) #AUC
