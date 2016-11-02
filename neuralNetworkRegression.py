import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn import linear_model
from sklearn.utils import shuffle

from sklearn import neural_network

#import data from database
df = pd.read_csv('angouleme_data_cardiac01_Low_FOV20_Record__weight_dose_ang1_ang2_ept_nFrame.csv',header=None)

#put data to x and y vectors
numberOfLines = 8972

xWeight = df.iloc[:,0].values
xWeight = xWeight.reshape(len(xWeight),1)


xAng1 = df.iloc[:,2].values
xAng1 = xAng1.reshape(len(xAng1),1)

xAng2 = df.iloc[:,3].values
xAng2 = xAng2.reshape(len(xAng2),1)

xEPT = df.iloc[:,4].values
xEPT = xEPT.reshape(len(xEPT),1)

yDose = df.iloc[:,1].values
xNOfFrame = df.iloc[:,5].values
yDosePerFrame = yDose/xNOfFrame
#yDosePerFrame = yDosePerFrame.reshape(len(yDosePerFrame),1)

logYdose = np.log(yDose)


xOnes = np.ones(len(xWeight))
xOnes = xOnes.reshape(len(xOnes),1)


def neuralNetworkRegression(xTrain, yTrain, xVal, yVal, myAlpha):
  #Make regression on EPT only and log of dose per frame
  regNN = neural_network.MLPRegressor(hidden_layer_sizes=(10, 10, 10, 10, 10, 10, 10, 10 ), alpha = 7,learning_rate_init = 0.0000000015, random_state = 0, solver = 'lbfgs', shuffle = True)
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

  """
  # compute the learning curves
  errorTrainLearn = np.zeros(500)#len(xTrain))
  errorValidLearn = np.zeros(500)#len(xTrain))

  
  for i in range(1,500):#len(xTrain)): #too much computations with all dataset...
    regNN.fit(xTrain[:i], yTrain[:i])
    yPredictTrain = regNN.predict(xTrain[:i])
    errorTrainLearn[i] = np.mean(np.square(yPredictTrain-yTrain[:i]))
    yPredictVal = regNN.predict(xVal)
    errorValidLearn[i] = np.mean(np.square(yPredictVal-yVal))

#  plt.hist(relativeErrorValidLearn,50)
#  plt.show()



  #Plot the learning curves
  learningIt = np.arange(0,500)#len(xTrain))

  plt.plot(learningIt, errorTrainLearn, learningIt, errorValidLearn)
  plt.show()
  #Should show model too simple cause of high bias => need to use more features
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

#X = np.concatenate((xWeight, np.square(xWeight), xAng1, np.square(xAng1), xAng2, np.square(xAng2)), axis=1)
X = np.concatenate((xWeight, xAng1, xAng2), axis=1)

X, yDosePerFrame = shuffle(X, yDosePerFrame, random_state = 0)


Xtrain = X[0:6000]
Xval = X[6001:8000]
Xtest = X[8001:]

Ytrain = yDosePerFrame[0:6000]
Yval = yDosePerFrame[6001:8000]
Ytest = yDosePerFrame[8001:]


neuralNetworkRegression(Xtrain,Ytrain,Xval,Yval, 0.001);
##mean relative error of 57 % => too much



