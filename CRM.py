import pandas as pd
import seaborn as sns
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier
import time 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc


'''
Read the data
'''

#read the data and drop useless features
data = pd.read_csv('original_data.csv')
data = data.drop('Unnamed: 0', axis = 1)

#encode the target value
le = LabelEncoder()
data['y']= le.fit_transform(data['y'])

'''
Find the outlier of the dataset and figure out the boundary
'''

#plot the box plot
f1 = sns.boxplot(data['duration'])
f1.figure.set_size_inches(8,5)
plt.show()

#figure out the outlier boundary

q3 = np.percentile(data['duration'], [75])
q1 = np.percentile(data['duration'], [25])
iqr = q3 - q1

print("Upper_bound:", q3 + 1.5*iqr)
print("Lower_bound:", q1 - 1.5*iqr)

#plot of feature distribution
sns.pairplot(data.iloc[:,:15])

'''
The correlation map
'''
corr = data.corr()
plt.figure(figsize=(15, 15)) 
sns.heatmap(corr, square=True, linewidths=.5, annot=True, fmt='.2f', cmap='coolwarm')


'''
Check and handle missing values
'''
#loop the data to count the number of missing values, which treated as unknown
missing_dict = {}
for col in data.columns:
    missing_dict[col] = 0
    for item in data[col]:
        if item == 'unknown':
            missing_dict[col] += 1

#drop the useless column
udata = data.drop(['Unnamed: 0', 'duration', 'cons.price.idx', 'euribor3m', 'nr.employed'], axis = 1)

# encoding the categorical data,
le1 = LabelEncoder()
for label in udata.columns:
    udata[label]= le1.fit_transform(udata[label])

# scaling the numerical data
for col in ['campaign', 'pdays', 'emp.var.rate', 'cons.conf.idx']:
    sc = StandardScaler(with_mean=True, with_std=True)
    udata[col] = sc.fit_transform(udata[col].values.reshape(-1,1))

# separate the data and target value
ux = udata.drop('y', axis = 1)
uy = udata['y']

# split the data into training and testing data by ratio 0.2
uX_train, uX_test, uY_train, uY_test = train_test_split(ux, uy, test_size = 0.2)

'''
Using decision tree or Naive Bayes to impute missing value
'''
#build and fit the decision tree
DT = DecisionTreeClassifier()
DT.fit(X_train, Y_train)

#build and fit the naive bayes
NB = GaussianNB()
NB.fit(X_train, Y_train)

#predict the result
Y_pred1 = NB.predict(X_test)
Y_pred2 = DT.predict(X_test)

#print the accuracy
print('accuracy of DT:', accuracy_score(Y_test, Y_pred2))
print('accuracy of NB:', accuracy_score(Y_test, Y_pred1))

#collect the missing value
marital_missing = []
marital_index = []
for i, row in pdata.iterrows():
    if row[2] == 3:
        marital_missing.append(row)
        marital_index.append(i)

#drop the row with missing values
marital_data = pdata.drop(marital_index)

#rebuilt the data set for predicting missing values
marital_test = pd.DataFrame(marital_missing, columns = col)
marital_test = marital_test.drop('marital', axis = 1)
marital_x = marital_data.drop('marital', axis = 1)
marital_y = marital_data['marital']

#split the new data set
X_train, X_test, Y_train, Y_test = train_test_split(marital_x, marital_y, test_size = 0.2)

#impute the missing value by decision treeand naive bayes
DT = DecisionTreeClassifier()
DT.fit(X_train, Y_train)

NB = GaussianNB()
NB.fit(X_train, Y_train)

Y_pred1 = NB.predict(X_test)
Y_pred2 = DT.predict(X_test)

#print the score after imputing
print('accuracy of DT:', accuracy_score(Y_test, Y_pred2))
print('accuracy of NB:', accuracy_score(Y_test, Y_pred1))


#predict the marital missing value
NB = GaussianNB()
NB.fit(marital_x, marital_y)

Y_pred3 = NB.predict(marital_test)
marital_test['marital'] = Y_pred3
pdata = pd.concat([marital_data, marital_test])

'''
Drop the missing value
'''

#get the index of missing value
missing_list = []
for i in range(len(mdata.index)):
    if 'unknown' in mdata.iloc[i].values:
        missing_list.append(i)

#drop the columns
mdata = mdata.drop(missing_list)
mdata = mdata.drop(['Unnamed: 0','duration','cons.price.idx', 'euribor3m', 'nr.employed', 'poutcome'], axis = 1)

'''
Accuracy of unknown policy without using PCA
'''

start_time1 = time.time()

#the MLPNN Classifier
uml = MLPClassifier(activation = 'logistic', learning_rate = 'adaptive', max_iter = 1000, verbose = False)
uml.fit(uX_train, uY_train)
uy1 = uml.predict(uX_test)
au1 = accuracy_score(uY_test, uy1)
print('accuracy of MLPNN:', au1)

#the Naive Bayes Classifier
unb = GaussianNB()
unb.fit(uX_train, uY_train)
uy2 = unb.predict(uX_test)
au2 = accuracy_score(uY_test, uy2) 
print('accuracy of NB:', au2)

#the SVM Classifier
usvm = SVC()
usvm.fit(uX_train, uY_train)
uy3 = usvm.predict(uX_test)
au3 = accuracy_score(uY_test, uy3)
print('accuracy of SVM:',au3)

#the KNN Classifier
uknn = KNeighborsClassifier(n_neighbors=5)
uknn.fit(uX_train, uY_train)
uy4 = uknn.predict(uX_test)
au4 = accuracy_score(uY_test, uy4)
print('accuracy of KNN:', au4)

#the Decision Tree Classifier
udt = DecisionTreeClassifier()
udt.fit(uX_train, uY_train)
uy5 = udt.predict(uX_test)
au5 = accuracy_score(uY_test, uy5) 
print('accuracy of DT:', au5)

ut = time.time() - start_time1

'''
Accuracy of prediction policy without using PCA
'''

start_time2 = time.time()

#the MLPNN Classifier
pml = MLPClassifier(activation = 'logistic', learning_rate = 'adaptive', max_iter = 1000, verbose = False)
pml.fit(pX_train, pY_train)
py1 = pml.predict(pX_test)
ap1 = accuracy_score(pY_test, py1)
print('accuracy of MLPNN:', ap1)

#the Naive Bayes Classifier
pnb = GaussianNB()
pnb.fit(pX_train, pY_train)
py2 = pnb.predict(pX_test)
ap2 = accuracy_score(pY_test, py2) 
print('accuracy of NB:', ap2)

#the SVM Classifier
psvm = SVC()
psvm.fit(pX_train, pY_train)
py3 = psvm.predict(pX_test)
ap3 = accuracy_score(pY_test, py3)
print('accuracy of SVM:',ap3)

#the KNN Classifier
pknn = KNeighborsClassifier(n_neighbors=5)
pknn.fit(pX_train, pY_train)
py4 = uknn.predict(pX_test)
ap4 = accuracy_score(pY_test, py4)
print('accuracy of KNN:', ap4)

#the Decision Tree Classifier
pdt = DecisionTreeClassifier()
pdt.fit(pX_train, pY_train)
py5 = pdt.predict(pX_test)
ap5 = accuracy_score(pY_test, py5) 
print('accuracy of DT:', ap5)

pt = time.time() - start_time2

'''
Accuracy of drop  policy without using PCA
'''

start_time3 = time.time()

#the MLPNN Classifier
mml = MLPClassifier(activation = 'logistic', learning_rate = 'adaptive', max_iter = 1000, verbose = False)
mml.fit(mX_train, mY_train)
my1 = mml.predict(mX_test)
am1 = accuracy_score(mY_test, my1)
print('accuracy of MLPNN:', am1)

#the Naive Bayes Classifier
mnb = GaussianNB()
mnb.fit(mX_train, mY_train)
my2 = mnb.predict(mX_test)
am2 = accuracy_score(mY_test, my2) 
print('accuracy of NB:', am2)

#the SVM Classifier
msvm = SVC()
msvm.fit(mX_train, mY_train)
my3 = msvm.predict(mX_test)
am3 = accuracy_score(mY_test, my3)
print('accuracy of SVM:',am3)

#the KNN Classifier
mknn = KNeighborsClassifier(n_neighbors=5)
mknn.fit(mX_train, mY_train)
my4 = mknn.predict(mX_test)
am4 = accuracy_score(mY_test, my4)
print('accuracy of KNN:', am4)

#the Decision Tree Classifier
mdt = DecisionTreeClassifier()
mdt.fit(mX_train, mY_train)
my5 = mdt.predict(mX_test)
am5 = accuracy_score(mY_test, my5) 
print('accuracy of DT:', am5)

mt = time.time() - start_time3

'''
Accuracy plot of three policy without using PCA
'''
x = [1, 2, 3 ,4 ,5 ]
y1 = [au1, au2, au3, au4, au5]
y2 = [ap1, ap2, ap3, ap4, ap5]
y3 = [am1, am2, am3, am4, am5]

#plot the figure of three policy
plt.figure(figsize = (8, 5))
plt.plot(x,y1,"-",x,y1,"ro",linewidth=1, label = 'Unknown') 
plt.plot(x,y2,"--", x,y2,"bo",linewidth=1, label = 'Predict') 
plt.plot(x,y3,"-.",x,y3,"go",linewidth=1, label = 'Drop') 
plt.xlabel("N model") 
plt.ylabel("Accuracy") 
plt.title("MLPNN -> NB -> SVM -> KNN -> DT")  
plt.legend()
plt.show()

#From the comparison result, we can see using 'unknown' to replace missing value can get the highest result, however, using prediction method, it has the lowest deviation between each model finally, directly dropping missing value cause the lowest accuracy

'''
Accuracy plot of three policy with using PCA
'''

import matplotlib.pyplot as plt

x = [1, 2, 3 ,4 ,5 ]
y11 = [au11, au21, au31, au41, au51]
y21 = [ap11, ap21, ap31, ap41, ap51]
y31 = [am11, am21, am31, am41, am51]

plt.figure(figsize = (8, 5))
plt.plot(x,y11,"-",x,y11,"ro",linewidth=1, label = 'Unknown') 
plt.plot(x,y21,"--", x,y21,"bo",linewidth=1, label = 'Predict') 
plt.plot(x,y31,"-.",x,y31,"go",linewidth=1, label = 'Drop') 
plt.xlabel("N model (PCA)")   
plt.ylabel("Accuracy")   
plt.title("MLPNN -> NB -> SVM -> KNN -> DT")   
plt.legend()
plt.show()


'''
PCA comparison
'''

#plot the unknown policy comparison
x1 = [1, 2, 3, 4, 5]
uwp = [au1, au2, au3, au4, au5]
up = [au11, au21, au31, au41, au51]

plt.figure(figsize = (7, 5))
plt.plot(x1,uwp,"-",x1,uwp,"ro",linewidth=1, label = 'without PCA') 
plt.plot(x1,up,"--", x1,up,"bo",linewidth=1, label = 'PCA') 

plt.xlabel("N model (unknown policy)")  
plt.ylabel("Accuracy")  
plt.title("MLPNN -> NB -> SVM -> KNN -> DT") 
plt.legend()
plt.show()

#plot the prediciton policy comparison
x2 = [1, 2, 3, 4, 5]
pwp = [ap1, ap2, ap3, ap4, ap5]
pp = [ap11, ap21, ap31, ap41, ap51]

plt.figure(figsize = (7, 5))
plt.plot(x2,pwp,"-",x2,pwp,"ro",linewidth=1, label = 'without PCA') 
plt.plot(x2,pp,"--", x2,pp,"bo",linewidth=1, label = 'PCA') 

plt.xlabel("N model (Prediction policy)") 
plt.ylabel("Accuracy")  
plt.title("MLPNN -> NB -> SVM -> KNN -> DT")
plt.legend()
plt.show()

#plot the drop policy comparison

x3 = [1, 2, 3, 4, 5]
mwp = [am1, am2, am3, am4, am5]
mp = [am11, am21, am31, am41, am51]

plt.figure(figsize = (7, 5))
plt.plot(x3,mwp,"-",x3,mwp,"ro",linewidth=1, label = 'without PCA') 
plt.plot(x3,mp,"--", x3,mp,"bo",linewidth=1, label = 'PCA') 

plt.xlabel("N model (Drop policy)")  
plt.ylabel("Accuracy")   
plt.title("MLPNN -> NB -> SVM -> KNN -> DT") 
plt.legend()
plt.show()

'''
ROC
'''

ML1_score = uml.predict_proba(uX_test)
NB1_score = unb.predict_proba(uX_test)
SVM1_score = usvm.decision_function(uX_test)
KNN1_score = uknn.predict_proba(uX_test)
DT1_score = udt.predict_proba(uX_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(uY_test, ML1_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
lw = 1
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MLPNN-ROC')
plt.legend(loc="lower right")
plt.show()

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(uY_test, NB1_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
lw = 1
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('NB-ROC')
plt.legend(loc="lower right")
plt.show()

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(uY_test, SVM1_score)
    roc_auc[i] = auc(fpr[i], tpr[i])

lw = 1
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM-ROC')
plt.legend(loc="lower right")
plt.show()

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(uY_test, KNN1_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

lw = 1
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN-ROC')
plt.legend(loc="lower right")
plt.show()

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(uY_test, DT1_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

lw = 1
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DT-ROC')
plt.legend(loc="lower right")
plt.show()