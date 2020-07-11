#load the database
from sklearn.datasets import load_iris
iris=load_iris()
print(iris)

#divide the dataset into 2 parts
#first path is data
#second part is target

x=iris.data
y=iris.target

print(x)
print(x.shape)#(150,4)
print(y)#target of 150 flowers
print(y.shape)#(150,)

############################################
#split the dataset into 2 parts ie for training and testing
#training---->80%
#testing----->20%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

print(x_train.shape)#(120,4)
print(y_train.shape)#(120,)
print(x_test.shape)#(30,4)
print(y_test.shape)#(30,)

############################################
#create a model KNN(K nearest neighbour)
from sklearn.neighbors import KNeighborsClassifier
K=KNeighborsClassifier(n_neighbors=5)

#train the model by using training dataset
K.fit(x_train,y_train)

#test the model
y_pred_knn=K.predict(x_test)

############################################
#find accuracy
from sklearn.metrics import accuracy_score
acc_knn=accuracy_score(y_test,y_pred_knn)
acc_knn=round(acc_knn*100,2)
print("accuracy in KNN model is :",acc_knn,"%")

############################################
#make prediction for a new flower
print(K.predict([[4,3,2,3]]))

#############################################
#implement Logistic Regression
from sklearn.linear_model import LogisticRegression
L=LogisticRegression(solver='liblinear',multi_class='auto')

#train the model
L.fit(x_train,y_train)
#test the model
y_pred_lg=L.predict(x_test)
#find accuracy
from sklearn.metrics import accuracy_score
acc_lg=accuracy_score(y_test,y_pred_lg)

acc_lg=round(acc_lg*100,2)
print("accuracy in logistic regression is",acc_lg,"%")

#################################################
#IMPLEMENT naive bayes algorithm
from sklearn.naive_bayes import GaussianNB
N=GaussianNB()

#train the model
N.fit(x_train,y_train)

#test the model by testing data
y_pred_nb=N.predict(x_test)

#find accuracy
from sklearn.metrics import accuracy_score
acc_nb=accuracy_score(y_test,y_pred_nb)
acc_nb=round(acc_nb*100,2)
print("accuracy score in Naive Bayes is",acc_nb,"%")

###################################################
#IMPLEMENT decision tree
from sklearn.tree import DecisionTreeClassifier

D=DecisionTreeClassifier()#its an object of DecisionTreeClassifier class

#train the model
D.fit(x_train,y_train)

#test the model
y_pred_dt=D.predict(x_test)

#find accuracy
from sklearn.metrics import accuracy_score
acc_dt=accuracy_score(y_test,y_pred_dt)
acc_dt=round(acc_dt*100,2)
print("accuracy in decision tree is",acc_dt,"%")













































