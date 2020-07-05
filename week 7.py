#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing dataset
dataset = pd.read_csv("C:/Users/jites/Desktop/datasets_4458_8204_winequality-red.csv")

#Info about dataset
print("data info: ", dataset.info())
print(dataset.head())

#EDA and Visualization

#Count of target variable
sns.countplot(x='quality', data=dataset)
plt.show()

#Plotting barplots for each column against target variable
fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='fixed acidity', data=dataset)
plt.show()

fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='volatile acidity', data=dataset)
plt.show()

fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='citric acid', data=dataset)
plt.show()

fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='residual sugar', data=dataset)
plt.show()

fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='chlorides', data=dataset)
plt.show()

fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='free sulfur dioxide', data=dataset)
plt.show()

fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='total sulfur dioxide', data=dataset)
plt.show()

fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='density', data=dataset)
plt.show()

fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='pH', data=dataset)
plt.show()

fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='sulphates', data=dataset)
plt.show()

fig = plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='alcohol', data=dataset)
plt.show()

#Data Preprocessing

#Making binary classifications for the target variable
#Dividing wine as good or bad by giving the limit for the quality
#Setting cutoffs for quality:
# 1 - Good, 0 - Bad

#Create an empty list reviews
reviews = []
for i in dataset['quality']:
    if i>=7:
        reviews.append('1')
    else:
        reviews.append('0')
dataset['Reviews'] = reviews

#Splitting X and y variables
X = dataset.iloc[:, :11].values
y = dataset['Reviews'].values

#Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

from sklearn.metrics import accuracy_score

#Fitting logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression accuracy score: ", accuracy_score(y_test, y_pred_lr))

#Fitting KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=100, metric='minkowski', p=2)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN accuracy score: ", accuracy_score(y_test, y_pred_knn))

#fitting Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy', random_state=0)
dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)
print("Decision Tree accuracy score: ", accuracy_score(y_test, y_pred_dtc))

#Fitting Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
print("Random Forest accuracy score: ", accuracy_score(y_test, y_pred_rfc))

#Fitting SVM
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM accuracy score: ", accuracy_score(y_test, y_pred_svm))

#Fitting Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("Naive Bayes accuracy score: ", accuracy_score(y_test, y_pred_nb))



