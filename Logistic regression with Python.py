# Importing packages

import pandas as pd # data processing
import numpy as np # working with arrays
import itertools 
import matplotlib.pyplot as plt # visualizations
from matplotlib import rcParams # plot size customization
from termcolor import colored as cl # text customization
from sklearn.model_selection import train_test_split # splitting the data
from sklearn.linear_model import LogisticRegression # model algorithm
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.metrics import jaccard_similarity_score as jss # evaluation metric
from sklearn.metrics import precision_score # evaluation metric
from sklearn.metrics import classification_report # evaluation metric
from sklearn.metrics import confusion_matrix # evaluation metric
from sklearn.metrics import log_loss # evaluation metric

rcParams['figure.figsize'] = (20, 10)

# Importing the data and EDA

df = pd.read_csv('tele_customer_data.csv')
df.drop(['Unnamed: 0', 'loglong', 'callwait', 'logtoll', 'voice', 'ebill', 'lninc', 'confer', 'internet', 'cardten', 
         'pager', 'longten', 'wiremon', 'equipmon', 'tollten', 'wireless', 'callcard', 'cardmon', 'tollmon'], 
        axis = 1, inplace = True)

for i  in df.columns:
    df[i] = df[i].astype(int)

df.head()

df.describe()

df.info()

# Splitting the data

X_var = np.asarray(df[['tenure', 'age', 'income', 'ed', 'employ', 'longmon', 'custcat']])
y_var = np.asarray(df['churn'])

print(cl('X_var samples : ', attrs = ['bold']), X_var[:5])
print(cl('y_var samples : ', attrs = ['bold']), y_var[:5])

X_var = StandardScaler().fit(X_var).transform(X_var)

print(cl(X_var[:5], attrs = ['bold']))

X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.3, random_state = 4)

print(cl('X_train samples : ', attrs = ['bold']), X_train[:5])
print(cl('X_test samples : ', attrs = ['bold']), X_test[:5])
print(cl('y_train samples : ', attrs = ['bold']), y_train[:10])
print(cl('y_test samples : ', attrs = ['bold']), y_test[:10])

# Modelling

lr = LogisticRegression(C = 0.1, solver = 'liblinear')
lr.fit(X_train,y_train)

yhat = lr.predict(X_test)
yhat_prob = lr.predict_proba(X_test)

print(cl('yhat samples : ', attrs = ['bold']), yhat[:10])
print(cl('yhat_prob samples : ', attrs = ['bold']), yhat_prob[:10])

# Evaluation

# 1. Jaccard Index

print(cl('Jaccard Similarity Score of our model is {}'.format(jss(y_test, yhat).round(2)), attrs = ['bold']))

# 2. Precision Score

print(cl('Precision Score of our model is {}'.format(precision_score(y_test, yhat).round(2)), attrs = ['bold']))

# 3. Log loss

print(cl('Log Loss of our model is {}'.format(log_loss(y_test, yhat).round(2)), attrs = ['bold']))

# 4. Classificaton report

print(cl(classification_report(y_test, yhat), attrs = ['bold']))

def plot_confusion_matrix(cm, classes,normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title, fontsize = 22)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45, fontsize = 13)
    plt.yticks(tick_marks, classes, fontsize = 13)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 fontsize = 15,
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label', fontsize = 16)
    plt.xlabel('Predicted label', fontsize = 16)

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, yhat, labels = [1,0])
np.set_printoptions(precision = 2)


# Plot non-normalized confusion matrix

plt.figure()
plot_confusion_matrix(cnf_matrix, classes = ['churn=1','churn=0'], normalize = False,  title = 'Confusion matrix')
plt.savefig('confusion_matrix.png')
