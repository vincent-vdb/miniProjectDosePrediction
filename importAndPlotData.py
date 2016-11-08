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
#xEPT = xEPT.reshape(len(xEPT),1)

xNOfFrame = df.iloc[:,13].values

yDosePerFrame = yDose/xNOfFrame
#yDosePerFrame = yDosePerFrame.reshape(len(yDosePerFrame),1)


plt.plot(xEPT, yDose,'x')
plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xEPT, yDose, xNOfFrame)

ax.set_xlabel('EPT')
ax.set_ylabel('Dose')
ax.set_zlabel('Nb of frame')

plt.show()

