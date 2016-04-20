import numpy as np

#import file
csv=open("data.csv")
data=np.loadtxt(csv,delimiter=',')
print data

X=data[:,9:30]
y=data[:,41]

print X
print y

#knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
knn=KNeighborsClassifier(n_neighbors=20)
knnresult=cross_val_score(knn,X,y,cv=55)
print ("knn avg: ",knnresult.mean())

#bagging tuned
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
bagging = BaggingClassifier(GradientBoostingClassifier(),max_samples=0.5, max_features=0.5)
baggingresult=cross_val_score(bagging,X,y,cv=55)
print ("bagging mean result:",baggingresult.mean())

#RF tuned
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=30,criterion='entropy',max_features=10,max_depth=6)
RFresult=cross_val_score(RF,X,y,cv=55)
#print ("RF result array:",RFresult)
print ("RF mean result:",RFresult.mean())

#adaboost tuned
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
ada=AdaBoostClassifier(n_estimators=500,learning_rate=0.1)
scores = cross_val_score(ada, X, y,cv=55)
adaResult=scores.mean()
print ("Adaboost mean result",adaResult.mean())

#gradient boosting(tuned)
from sklearn.ensemble import GradientBoostingClassifier
gradientClf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1,max_depth=1, random_state=0)
gradientResult=cross_val_score(gradientClf,X,y,cv=55)
print ("Gradient mean result",gradientResult.mean())

#svm(linear vs rbf)
from sklearn import svm
rbfsvmclf=svm.SVC()
rbfsvmresult=cross_val_score(rbfsvmclf,X,y,cv=55)
linearsvmclf=svm.SVC(kernel='linear')
linearsvmresult=cross_val_score(linearsvmclf,X,y,cv=55)
print ("radial basis function kernel svm avg:", rbfsvmresult.mean())
print ("linear kernel svm avg:", linearsvmresult.mean())