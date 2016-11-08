import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import preprocessing

#import data from database
df = pd.read_csv('data_cardiac01_low_fov20_record_moreFeatures.csv',header=None)

#put data to x and y vectors
xWeight = df.iloc[:,0].values
xWeight = xWeight.reshape(len(xWeight),1)

xSID = df.iloc[:,1].values
xSID = xSID.reshape(len(xSID),1)

xTableHeight = df.iloc[:,2].values
xTableHeight = xTableHeight.reshape(len(xTableHeight),1)

xSF = df.iloc[:,3].values
xSF = xSF.reshape(len(xSF),1)

xFS = df.iloc[:,4].values
xFS = xFS.reshape(len(xFS),1)

xKV = df.iloc[:,5].values
xKV = xKV.reshape(len(xKV),1)

xPW = df.iloc[:,6].values
xPW = xPW.reshape(len(xPW),1)

xMAS = df.iloc[:,7].values
xMAS = xMAS.reshape(len(xMAS),1)

xTED = df.iloc[:,8].values
xTED = xTED.reshape(len(xTED),1)

yDose = df.iloc[:,9].values

xAng1 = df.iloc[:,10].values
xAng1 = xAng1.reshape(len(xAng1),1)

xAng2 = df.iloc[:,11].values
xAng2 = xAng2.reshape(len(xAng2),1)

xEPT = df.iloc[:,12].values
xEPT = xEPT.reshape(len(xEPT),1)

xNOfFrame = df.iloc[:,13].values

yDosePerFrame = yDose/xNOfFrame
yDosePerFrame = yDosePerFrame.reshape(len(yDosePerFrame),1)


def regressionAndLearningCurve(xTrain, yTrain, xVal, yVal, myAlpha):
  #Make regression on EPT only and log of dose per frame
  reg = linear_model.Ridge(alpha = myAlpha)
  reg.fit(xTrain, yTrain)

  #Compute the error on the training set and on the validation set
  yPredictTrain = reg.predict(xTrain)
  errorTrain = np.mean(np.square(yPredictTrain-yTrain))

  yPredictVal = reg.predict(xVal)
  errorValid = np.mean(np.square(yPredictVal-yVal))

  print("error on training set :",errorTrain)
  print("error on valid set :",errorValid)
  print(reg.coef_)

  print(np.mean(yTrain))
  print(np.mean(yVal))

  relativeErrorTrainLearn = np.abs((yPredictTrain - yTrain)/yTrain)
  relativeErrorValidLearn = np.abs((yPredictVal - yVal)/yVal)
  print("mean train relative error: ",np.mean(relativeErrorTrainLearn))
  print("mean valid relative error: ",np.mean(relativeErrorValidLearn))
  print("std valid relative error: ", np.std(relativeErrorValidLearn))

  
  # compute the learning curves
  errorTrainLearn = np.zeros(len(xTrain))
  errorValidLearn = np.zeros(len(xTrain))

  for i in range(1,len(xTrain)):
    reg.fit(xTrain[:i], yTrain[:i])
    yPredictTrain = reg.predict(xTrain[:i])
    errorTrainLearn[i] = np.mean(np.square(yPredictTrain-yTrain[:i]))
    yPredictVal = reg.predict(xVal)
    errorValidLearn[i] = np.mean(np.square(yPredictVal-yVal))


#  plt.hist(relativeErrorValidLearn,50)
#  plt.show()



  #Plot the learning curves
  learningIt = np.arange(0,len(xTrain))

  plt.plot(learningIt, errorTrainLearn, learningIt, errorValidLearn)
  plt.show()
  #Should show model too simple cause of high bias => need to use more features
  

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
#mean relative error 87 %

#X = np.concatenate((xWeight, np.square(xWeight), xSID, np.square(xSID), xTableHeight, np.square(xTableHeight), xAng1, np.square(xAng1), xAng2, np.square(xAng2)), axis=1)
#mean relative error 85 %

#X = np.concatenate((xWeight, np.square(xWeight), np.exp(xWeight), xSID, np.square(xSID), xTableHeight, np.square(xTableHeight), xAng1, np.square(xAng1), xAng2, np.square(xAng2), xKV, np.exp(xKV), xSF, np.exp(xSF), np.exp(-xSF)), axis=1)
#mean relative error 59 %

#X = np.concatenate((np.square(xEPT), xEPT,  xSID, np.square(xSID), xTableHeight, np.square(xTableHeight), xKV, np.exp(xKV), xSF, np.exp(xSF), np.exp(-xSF)), axis=1)
#mean relative error 24 % +- 40 %

#X = np.concatenate(( np.exp(-xSF), np.square(xSID), xWeight, np.square(xWeight), xWeight*np.sin(xAng1), xWeight*np.sin(xAng2), xKV, np.exp(xKV)), axis=1)
#mean relative error 59 % +- 79 %


xAng1 = xAng1/10*np.pi/180
xAng2 = xAng2/10*np.pi/180

xSF = preprocessing.scale(xSF)
xSID = preprocessing.scale(xSID)
xWeight = preprocessing.scale(xWeight)
xKV = preprocessing.scale(xKV)

"""
print(np.min(xAng1)) #-45 degree
print(np.max(xAng1)) #+39 degree

print(np.min(xAng2)) #-51 degree
print(np.max(xAng2)) #+90 degree
"""

X = np.concatenate(( np.exp(-xSF), np.square(xSID), xWeight, np.square(xWeight), np.exp(xWeight*np.sin(xAng1)), np.exp(xWeight*np.sin(xAng2)), xKV, np.square(xKV), np.exp(xKV)), axis=1)


X, yDosePerFrame = shuffle(np.exp(xEPT), yDosePerFrame, random_state = 0)


Xtrain = X[0:6000]
Xval = X[6001:8000]
Xtest = X[8001:]

Ytrain = yDosePerFrame[0:6000]
Yval = yDosePerFrame[6001:8000]
Ytest = yDosePerFrame[8001:]


regressionAndLearningCurve(Xtrain,Ytrain,Xval,Yval, 0.);





