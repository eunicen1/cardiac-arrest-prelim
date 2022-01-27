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
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MaxAbsScaler

np.random.seed(457)

#setup
#START added
#df = pd.read_csv('data2.csv', delimiter=',')
df = pd.read_csv('data2.csv', delimiter=',')
del df[df.columns[0]]
scaler = MaxAbsScaler()
scaler.fit(df)
scaled = scaler.transform(df)
df = pd.DataFrame(scaled, columns=df.columns)
#STOP added

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


kscore = 4


models = []
accuracies = []
# Training Model 1
nnModel = KNeighborsClassifier(n_neighbors=int(kscore))
models.append(("knnModel", nnModel))
nnModel.fit(Xtrain, ytrain)
ypred = nnModel.predict(Xtest)
acc = accuracy_score(ytest, ypred)
accuracies.append(acc)

# Training Model 2
dtModel = DecisionTreeClassifier()
models.append(("dtModel1", dtModel))
dtModel.fit(Xtrain, ytrain)
ypred = dtModel.predict(Xtest)
acc = accuracy_score(ytest, ypred)
accuracies.append(acc)

# Training Model 3
dtModel = DecisionTreeClassifier()
models.append(("dtModel2", dtModel))
dtModel.fit(Xtrain, ytrain)
ypred = dtModel.predict(Xtest)
acc = accuracy_score(ytest, ypred)
accuracies.append(acc)

# Training Model 4
dtModel = DecisionTreeClassifier()
models.append(("dtModel3", dtModel))
dtModel.fit(Xtrain, ytrain)
ypred = dtModel.predict(Xtest)
acc = accuracy_score(ytest, ypred)
accuracies.append(acc)

print("Model Accuracies", accuracies)

ensemble = VotingClassifier(models)
ec = ensemble.fit(Xtrain, ytrain)
pred = ec.predict(Xtest)
ens_acc = accuracy_score(pred, ytest)
ens_f1 = f1_score(pred, ytest),2
print("Ensemble accuracy is: ", ens_acc)
print("Ensemble f-1 score is: ", ens_f1)
print(confusion_matrix(pred, ytest))
print(classification_report(pred, ytest))
