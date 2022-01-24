import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve as pltroc
from sklearn.metrics import precision_recall_curve as pltprc
from sklearn.metrics import f1_score

np.random.seed(457)

#setup
df = pd.read_csv('data2.csv', delimiter=',')
df = df.sample(frac=1)
arr = df.to_numpy()
splitarr = np.array_split(arr, 5, axis=0)
train = np.concatenate((splitarr[0], splitarr[4]), axis=0)
valid = splitarr[1]
test = splitarr[2]
a, b = train.shape
c, d = valid.shape
e, f = test.shape
Xtrain = train[:,:-1]
ytrain = train[:,b-1]
Xvalid = valid[:,:-1]
yvalid = valid[:,d-1]
Xtest = test[:,:-1]
ytest = test[:,f-1]

#cross-validation for KNN
kscores = []
for k in range(1,c):
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    knnModel = KNeighborsClassifier(k)
    scores = cross_val_score(knnModel, Xvalid, yvalid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    kscores.append(np.mean(np.abs(scores)))

kscores = np.concatenate((np.arange(1,c).reshape(c-1,1), np.array(kscores).reshape(c-1,1)),axis=1)
print(kscores)
kscores = np.array([kscores[i,:] for i in range(c-1) if np.isnan(kscores[i,1]) == False])
print(kscores)
kscore = kscores[np.argmin(kscores[:,1]),0] #select best kscore
print('best score is @ seed === 457: ', kscore)
#training
nnModel = KNeighborsClassifier(n_neighbors=int(kscore))
nnModel.fit(Xtrain, ytrain)
# plot_classifier(nnModel, Xtrain,ytrain)
ypred = nnModel.predict(Xtest)
acc = accuracy_score(ytest, ypred)
cm = confusion_matrix(ytest, ypred)
#plot accuracy
pltroc(nnModel, Xtest, ytest)
plt.title("ROC Curve and AUC score with AUC: "+ str(round(acc,2)))
plt.show()
print(acc, cm)
#plot precision / recall
os = (ypred == ytest)
yTrue = np.zeros(ytest.shape)
yTrue[os] = 1
predprob = nnModel.predict_proba(Xtest)

prec, rec, _ = pltprc(y_true=ytest, probas_pred=predprob[:, 1])
f1 = str(round(f1_score(ypred, ytest),2))

plt.plot(rec, prec, marker='.', label='Logistic')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Precision and Recall Curve with f1: "+ f1)
plt.show()
