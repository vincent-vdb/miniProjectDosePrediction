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

xEPT = xEPT.reshape(len(xEPT),1)
yDosePerFrame = yDosePerFrame.reshape(len(yDosePerFrame),1)

reg = linear_model.LinearRegression()
reg.fit(xEPT, yDosePerFrame)

x = np.arange(0,40,1)
x = x.reshape(len(x),1)
yPredict = reg.predict(x)
plt.scatter(x,yPredict)


#from Perceptron import *
#from myObjectPerceptron import *

#from myObjectAdalineGD import *
#from AdalineGD import *

#import data from database
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
#df.tail


#put data to x and y vectors
#y = df.iloc[0:100,4].values
#y = np.where(y == 'Iris-setosa',-1,1)
#x = df.iloc[0:100,[0,2]].values

#x[:,0] = (x[:,0] - x[:,0].mean())/x[:,0].std()
#x[:,1] = (x[:,1] - x[:,1].mean())/x[:,1].std()

#plot data
"""
plt.scatter(x[:50,0], x[:50,1], color='red',marker='o',label='setosa')
plt.scatter(x[50:100,0], x[50:100,1], color='blue',marker='x',label='versicolor')

plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()
"""
"""
#ppn = Perceptron(0.1, 10)
ppn = AdalineGD(0.01, 10)
ppn.fit(x,y)

plt.plot(range(1,len(ppn._errors)+1),ppn._errors,marker='o')

plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()



def plotDecisionRegions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)





plotDecisionRegions(x, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('./perceptron_2.png', dpi=300)
plt.show()

"""

