# -*- coding: utf-8 -*-
"""
Created on Sun April 28 11:34:23 2019
@author: ryokugyu
"""
#For clearning all the pre existential variables in the spyder space
def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
clear_all()


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from pyspark.sql import SparkSession
import warnings
warnings.filterwarnings('ignore')

def X_connect():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    X = spark.read.option("header","false").csv("hdfs://localhost:9000//CBDSProject/X.csv")
    return X

def y_connect():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()    
    y =  spark.read.option("header","false").csv("hdfs://localhost:9000//CBDSProject/y.csv")
    return y


def mean_connect():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()    
    mean_imputed_dataframes =  spark.read.option("header","true").csv("hdfs://localhost:9000//CBDSProject/mean_imputed_dataframes.csv")
    return mean_imputed_dataframes
    
    
def knn_connect():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()    
    knn_imputed_dataframes = spark.read.option("header","true").csv("hdfs://localhost:9000//CBDSProject/knn_imputed_dataframes.csv")
    return knn_imputed_dataframes

def dataframes_connect():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()    
    dataframes = spark.read.option("header","true").csv("hdfs://localhost:9000//CBDSProject/tempfile.csv")
    return dataframes

X = X_connect()
y = y_connect()
mean_imputed_dataframes = mean_connect()
knn_imputed_dataframes = knn_connect()
dataframes = dataframes_connect()
X = X.toPandas()
y = y.toPandas()
mean_imputed_dataframes = mean_imputed_dataframes.toPandas()
knn_imputed_dataframes = knn_imputed_dataframes.toPandas()
dataframes = dataframes.toPandas()
#print(X.show())
#print(y.show())
#print(mean_imputed_dataframes.show())
#print(knn_imputed_dataframes.show())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Supported Vector Machine model
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train.values.ravel())
predictions = model.predict(X_test)
accuracy=accuracy_score(y_test, predictions)
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train.values.ravel())
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print('Accuracy of SVM:',accuracy)

#Decision Tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train.values.ravel())
predict_dtree = dtree.predict(X_test)
accuracy=accuracy_score(y_test, predict_dtree)
print('Accuracy of Decision Tree:',accuracy)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train.values.ravel())
rfc_pred = rfc.predict(X_test)
accuracy=accuracy_score(y_test,rfc_pred)
print('Accuracy of Random Forest:', accuracy)

# K-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train.values.ravel())
k_pred = knn.predict(X_test)
accuracy=accuracy_score(y_test,k_pred)
print('Accuracy of KNN:', accuracy)
# -----------------------------------------------------------------------------
X = knn_imputed_dataframes[["X1", "X2", "X6", "X7","X9","X10", "X34"]]
y = dataframes['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.svm import SVC

# Supported Vector Machine model
model = SVC()
model.fit(X_train,y_train.values.ravel())
predictions = model.predict(X_test)
accuracy=accuracy_score(y_test, predictions)
print('Accuracy of SVM with KNN Imputed Mean:',accuracy)


# K-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train.values.ravel())
k_pred = knn.predict(X_test)
accuracy=accuracy_score(y_test,k_pred)
print('Accuracy of KNN with KNN Imputed Mean:', accuracy)


X = knn_imputed_dataframes[["X1", "X2", "X6", "X7","X9","X10", "X34"]]
y = dataframes['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Decision Tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train.values.ravel())
predict_dtree = dtree.predict(X_test)
accuracy=accuracy_score(y_test, predict_dtree)
print('Accuracy of Decision tree with KNN Imputed Mean',accuracy)
#print(confusion_matrix(y_test,predict_dtree))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train.values.ravel())
rfc_pred = rfc.predict(X_test)
accuracy=accuracy_score(y_test,rfc_pred)
print('Accuracy of Random Forest with KNN Imputed Mean', accuracy)



