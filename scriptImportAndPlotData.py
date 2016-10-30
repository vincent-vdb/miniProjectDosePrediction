import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn import linear_model

#import data from database
df = pd.read_csv('angouleme_data_cardiac01_Low_FOV20_Record__weight_dose_ang1_ang2_ept_nFrame.csv',header=None)

#put data to x and y vectors
numberOfLines = 8972

xWeight = df.iloc[:numberOfLines,0].values
xWeight = xWeight.reshape(len(xWeight),1)


xAng1 = df.iloc[:numberOfLines,2].values
xAng1 = xAng1.reshape(len(xAng1),1)

xAng2 = df.iloc[:numberOfLines,3].values
xAng2 = xAng2.reshape(len(xAng2),1)

xEPT = df.iloc[:numberOfLines,4].values
xEPT = xEPT.reshape(len(xEPT),1)

yDose = df.iloc[:numberOfLines,1].values
xNOfFrame = df.iloc[:numberOfLines,5].values
yDosePerFrame = yDose/xNOfFrame
yDosePerFrame[np.where(yDosePerFrame==0.0)]=0.001
yDosePerFrame = yDosePerFrame.reshape(len(yDosePerFrame),1)

logYdose = np.log(yDose)


xOnes = np.ones(len(xWeight))
xOnes = xOnes.reshape(len(xOnes),1)


X = np.concatenate((xOnes, xWeight, xAng1, xAng2), axis=1)
#X = np.concatenate((xOnes,xEPT),axis=1)


#Create train, valid and test datasets
Xtrain = X[0:6000]
Xval = X[6001:8000]
Xtest = X[8000:]

Ytrain = yDosePerFrame[0:6000]
Yval = yDosePerFrame[6001:8000]
Ytest = yDosePerFrame[8000:]



#plt.hist(xWeight,50)
#plt.show()

#plot data
plt.scatter(xEPT[:200], np.log(yDosePerFrame[:200]), color='red',marker='o',label='dose')
plt.xlabel('EPT (cm)');
plt.ylabel('log of dose per frame (log(Gy))');
#plt.scatter(x[50:100,0], x[50:100,1], color='blue',marker='x',label='versicolor')
plt.show()


#Fix temporaire pour eliminer la dose nulle

reg = linear_model.LinearRegression()
reg.fit(xEPT, np.log(yDosePerFrame))

print(reg.coef_)


xfit = np.arange(0,40,1)
xfit = xfit.reshape(len(xfit),1)
#xfit = np.concatenate(((np.ones(40).reshape(40,1)), xfit),axis=1)
yPredict = reg.predict(xfit)
plt.plot(xfit, yPredict, 'bs',xEPT, np.log(yDosePerFrame),'rx')
plt.show()


