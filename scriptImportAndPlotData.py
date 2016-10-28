import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


#import data from database
#df = pd.read_csv('angouleme_data_1jan2016_12october2016_selectedForProject.csv',header=None)
df = pd.read_csv('angouleme_data_cardiac01_Low_FOV20_Record__weight_dose_ang1_ang2_ept_nFrame.csv',header=None)
#put data to x and y vectors
numberOfLines = 8972

xWeight = df.iloc[:numberOfLines,0].values
yDose = df.iloc[:numberOfLines,1].values
xAng1 = df.iloc[:numberOfLines,2].values
xAng2 = df.iloc[:numberOfLines,3].values
xEPT = df.iloc[:numberOfLines,4].values

xNOfFrame = df.iloc[:numberOfLines,5].values

logYdose = np.log(yDose)

yDosePerFrame = yDose/xNOfFrame

plt.hist(xWeight,50)
plt.show()

#plot data
plt.scatter(xEPT[:200], np.log(yDosePerFrame[:200]), color='red',marker='o',label='dose')
#plt.scatter(x[50:100,0], x[50:100,1], color='blue',marker='x',label='versicolor')
plt.show()

from sklearn import linear_model

#X = [xWeight, xAng1]

#Fix temporaire pour eliminer la dose nulle
yDosePerFrame[np.where(yDosePerFrame==0.0)]=0.001


xEPT = xEPT.reshape(len(xEPT),1)
yDosePerFrame = yDosePerFrame.reshape(len(yDosePerFrame),1)

reg = linear_model.LinearRegression()
reg.fit(xEPT, np.log(yDosePerFrame))

x = np.arange(0,40,1)
x = x.reshape(len(x),1)
yPredict = reg.predict(x)
plt.plot(x, yPredict, 'bs',xEPT, np.log(yDosePerFrame),'rx')
plt.show()


