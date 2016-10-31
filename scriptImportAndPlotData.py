import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn import linear_model

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
yDosePerFrame = yDosePerFrame.reshape(len(yDosePerFrame),1)

logYdose = np.log(yDose)


xOnes = np.ones(len(xWeight))
xOnes = xOnes.reshape(len(xOnes),1)


X = np.concatenate((xOnes, xWeight, xAng1, xAng2), axis=1)
#X = np.concatenate((xOnes,xEPT),axis=1)

#Create example with only EPT
Xepttrain = xEPT[0:6000]
Xeptval = xEPT[6001:8000]
Xepttest = xEPT[8001:]

#Create train, valid and test datasets
Xtrain = X[0:6000]
Xval = X[6001:8000]
Xtest = X[8001:]

Ytrain = yDosePerFrame[0:6000]
Yval = yDosePerFrame[6001:8000]
Ytest = yDosePerFrame[8001:]



#Make regression on EPT only and log of dose per frame
reg = linear_model.LinearRegression()
reg.fit(Xepttrain, np.log(Ytrain))

#Compute the error on the training set and on the validation set
yPredictTrain = reg.predict(Xepttrain)
errorTrain = 1/len(Xepttrain)*np.sum(np.square(yPredictTrain-np.log(Ytrain)))

yPredictVal = reg.predict(Xeptval)
errorValid = 1/len(Xeptval)*np.sum(np.square(yPredictVal-np.log(Yval)))

print("error on training set :",errorTrain)
print("error on valid set :",errorValid)

# compute the learning curves

errorTrainLearn = np.zeros(len(Xepttrain))
errorValidLearn = np.zeros(len(Xepttrain))

for i in range(1,len(Xepttrain)):
  reg.fit(Xepttrain[:i], np.log(Ytrain[:i]))
  yPredictTrain = reg.predict(Xepttrain[:i])
  errorTrainLearn[i] = 1/(i+1)*np.sum(np.square(yPredictTrain-np.log(Ytrain[:i])))
  yPredictVal = reg.predict(Xeptval)
  errorValidLearn[i] = 1/len(Xeptval)*np.sum(np.square(yPredictVal-np.log(Yval)))


#Plot the learning curves
learningIt = np.arange(0,len(Xepttrain))

plt.plot(learningIt, errorTrainLearn, learningIt, errorValidLearn)
plt.show()
#Should show model too simple cause of high bias => need to use more features


"""
#plot data
plt.scatter(xEPT[:200], np.log(yDosePerFrame[:200]), color='red',marker='o',label='dose')
plt.xlabel('EPT (cm)');
plt.ylabel('log of dose per frame (log(Gy))');
#plt.scatter(x[50:100,0], x[50:100,1], color='blue',marker='x',label='versicolor')
plt.show()


reg = linear_model.LinearRegression()
reg.fit(xEPT, np.log(yDosePerFrame))


print(reg.coef_)


xfit = np.arange(0,40,1)
xfit = xfit.reshape(len(xfit),1)
#xfit = np.concatenate(((np.ones(40).reshape(40,1)), xfit),axis=1)
yPredict = reg.predict(xfit)
plt.plot(xfit, yPredict, 'bs',xEPT, np.log(yDosePerFrame),'rx')
plt.show()
"""

