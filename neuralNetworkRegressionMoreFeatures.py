import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn import linear_model
from sklearn.utils import shuffle

from sklearn import preprocessing

from sklearn import neural_network

#import data from database
df = pd.read_csv('data_cardiac01_low_fov20_record_moreFeatures.csv',header=None)

#put data to x and y vectors
xWeight = df.iloc[:,0].values
xWeight = preprocessing.scale(xWeight.reshape(len(xWeight),1))

xSID = df.iloc[:,1].values
xSID = preprocessing.scale(xSID.reshape(len(xSID),1))

xTableHeight = df.iloc[:,2].values
xTableHeight = preprocessing.scale(xTableHeight.reshape(len(xTableHeight),1))

xSF = df.iloc[:,3].values
xSF = preprocessing.scale(xSF.reshape(len(xSF),1))

xFS = df.iloc[:,4].values
xFS = xFS.reshape(len(xFS),1)

xKV = df.iloc[:,5].values
xKV = preprocessing.scale(xKV.reshape(len(xKV),1))

xPW = df.iloc[:,6].values
xPW = preprocessing.scale(xPW.reshape(len(xPW),1))

xMAS = df.iloc[:,7].values
xMAS = preprocessing.scale(xMAS.reshape(len(xMAS),1))

xTED = df.iloc[:,8].values
xTED = preprocessing.scale(xTED.reshape(len(xTED),1))

yDose = df.iloc[:,9].values

xAng1 = df.iloc[:,10].values
xAng1 = preprocessing.scale(xAng1.reshape(len(xAng1),1))

xAng2 = df.iloc[:,11].values
xAng2 = preprocessing.scale(xAng2.reshape(len(xAng2),1))

xEPT = df.iloc[:,12].values
xEPT = preprocessing.scale(xEPT.reshape(len(xEPT),1))

xNOfFrame = df.iloc[:,13].values

yDosePerFrame = yDose/xNOfFrame
#yDosePerFrame = yDosePerFrame.reshape(len(yDosePerFrame),1)



def neuralNetworkRegression(xTrain, yTrain, xVal, yVal, myAlpha):
  #Make regression on EPT only and log of dose per frame
  regNN = neural_network.MLPRegressor(hidden_layer_sizes=(10, 10, 10, 10, 10), alpha = 0.,learning_rate_init = 0.0000000015, random_state = 0, solver = 'lbfgs', shuffle = True)#, batch_size = 500)
  regNN.fit(xTrain, yTrain)

  #Compute the error on the training set and on the validation set
  yPredictTrain = regNN.predict(xTrain)
  errorTrain = np.mean(np.square(yPredictTrain-yTrain))

  yPredictVal = regNN.predict(xVal)
  errorValid = np.mean(np.square(yPredictVal-yVal))

  print("error on training set :",errorTrain)
  print("error on valid set :",errorValid)
#  print(regNN.coef_)

  print(np.mean(yTrain))
  print(np.mean(yVal))

  relativeErrorTrainLearn = np.abs((yPredictTrain - yTrain)/yTrain)
  relativeErrorValidLearn = np.abs((yPredictVal - yVal)/yVal)
  print("mean train relative error: ",np.mean(relativeErrorTrainLearn))
  print("mean valid relative error: ",np.mean(relativeErrorValidLearn))

  
  # compute the learning curves
  errorTrainLearn = np.zeros(100)#len(xTrain))
  errorValidLearn = np.zeros(100)#len(xTrain))

  """
  for i in range(1,100):#len(xTrain)): #too much computations with all dataset...
    regNN.fit(xTrain[:50*i], yTrain[:50*i])
    yPredictTrain = regNN.predict(xTrain[:50*i])
    errorTrainLearn[i] = np.mean(np.square(yPredictTrain-yTrain[:50*i]))
    yPredictVal = regNN.predict(xVal)
    errorValidLearn[i] = np.mean(np.square(yPredictVal-yVal))

#  plt.hist(relativeErrorValidLearn,50)
#  plt.show()



  #Plot the learning curves
  learningIt = np.arange(0,100)#len(xTrain))

  plt.plot(learningIt, errorTrainLearn, learningIt, errorValidLearn)
  plt.show()
  """

  """
  plotx = np.arange(1,50)
  plotx = plotx.reshape(len(plotx),1)
  ploty = reg.predict(plotx)

  plt.plot(plotx, ploty, xTrain[:1000], yTrain[:1000], 'rx')
  plt.show()
  """

  return


"""
#Create example with only EPT
xEPT, yDosePerFrame = shuffle(xEPT, yDosePerFrame, random_state = 0)

Xepttrain = xEPT[0:6000]
Xeptval = xEPT[6001:8000]
Xepttest = xEPT[8001:]

Ytrain = yDosePerFrame[0:6000]
Yval = yDosePerFrame[6001:8000]
Ytest = yDosePerFrame[8001:]



# make the regression for only EPT
regressionAndLearningCurve(Xepttrain,np.log(Ytrain),Xeptval,np.log(Yval), 0);
# mean relative error of 19 %, high bias case
"""
#make regression on weight, weight squared angle 1 and 2

#X = np.concatenate((xWeight, xSID, xTableHeight, xAng1, xAng2), axis=1)
# 88 % with alpha 0.1 and 5 times 10 hidden units

X = np.concatenate((xWeight, xSID, xTableHeight, xKV, xTED, xAng1, xAng2), axis=1)
# 69 % with alpha 0.1 and 5 times 10 hidden units
# 53 % with alpha 0.1 and 5 times 10 hidden units and scaling



#X = np.concatenate((xWeight, xSID, xTableHeight, xKV, xTED, xAng1, xAng2), axis=1)



#X = np.concatenate((xWeight, xSID, xTableHeight, xKV, xTED, xEPT, xAng1, xAng2), axis=1)
# 16 % with alpha 0.1 and 5 times 10 hidden units and scaling

X, yDosePerFrame = shuffle(X, yDosePerFrame, random_state = 0)


Xtrain = X[0:6000]
Xval = X[6001:8000]
Xtest = X[8001:]

Ytrain = yDosePerFrame[0:6000]
Yval = yDosePerFrame[6001:8000]
Ytest = yDosePerFrame[8001:]


neuralNetworkRegression(Xtrain,Ytrain,Xval,Yval, 0.001);
##mean relative error of 57 % => too much



