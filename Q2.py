from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data=load_iris()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target

print("Dataset Info :")
print(X.describe())
print("Target Classes :\n",data.target_names)

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)


y_pred=knn.predict(X_test)

print("Accuracy without Scaling : ",accuracy_score(Y_test,y_pred))

scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)
X_train_scaled,X_test_scaled,Y_train_scaled,Y_test_scaled=train_test_split(X_scaled,y,test_size=0.2,random_state=42)
knn_scaled=KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled,Y_train_scaled)

y_pred_scored=knn_scaled.predict(X_test_scaled)

print("Accuracy with MinMax Scaling : ",accuracy_score(Y_test_scaled,y_pred_scored))