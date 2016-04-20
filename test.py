import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
#import file
csv=open("data.csv")
data=np.loadtxt(csv,delimiter=',')
#print data

X=data[:,9:30]
y=data[:,41]

from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
# standardize the data attributes
standardized_X = preprocessing.scale(X)
print
#knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
knn=KNeighborsClassifier(n_neighbors=15,weights='distance',algorithm='auto')
knnresult=cross_val_score(knn,X,y,cv=55)
print ("knn avg: ",knnresult.mean())
