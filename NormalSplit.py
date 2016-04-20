import numpy as np
from sklearn import metrics
#import file
csv=open("data.csv")
data=np.loadtxt(csv,delimiter=',')
print data

from sklearn.cross_validation import  train_test_split
train, test = train_test_split(data, test_size = 0.2)
trainx=train[:,9:30]
trainy=train[:,41]
testx=test[:,9:30]
testy=test[:,41]


X=data[:,9:30]
y=data[:,41]

print X
print y

#knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
knn=KNeighborsClassifier(n_neighbors=20)
knn.fit(trainx,trainy)
knnExp=testy
knnPred=knn.predict(testx)
print (metrics.classification_report(knnExp,knnPred))
#knnresult=cross_val_score(knn,X,y,cv=10)
#print ("knn avg: ",knnresult.mean())

#bagging
"""from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
baggingresult=cross_val_score(bagging,X,y,cv=20)
print ("bagging mean result:",baggingresult.mean())

#RF tuned
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=30,criterion='entropy',max_features=10,max_depth=6)
RFresult=cross_val_score(RF,X,y,cv=20)
#print ("RF result array:",RFresult)
print ("RF mean result:",RFresult.mean())

#adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
ada=AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(ada, X, y,cv=20)
adaResult=scores.mean()
print ("Adaboost mean result",adaResult.mean())

#gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
gradientClf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
gradientResult=cross_val_score(gradientClf,X,y,cv=20)
print ("Gradient mean result",gradientResult.mean())

#svm(Needs tune!!!!)
from sklearn import svm
svmclf=svm.SVC()
svmresult=cross_val_score(svmclf,X,y,cv=20)
print ("svm avg:", svmresult.mean())"""