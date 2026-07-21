import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv("bike_sharing_hourly.csv")

print("Dataset Info : ")
print(df.info())
print("Dataset Preview :")
print(df.head())

df['dteday']=pd.to_datetime(df["dteday"])
df['day_of_week']=pd.to_datetime(df["day_of_week"])
df['month']=pd.to_datetime(df["month"])
df['year']=pd.to_datetime(df["year"])

print("New features derived from date column")
print(df[['dteday','day_of_week','month','year']].head())

X=df['temp']
Y=df['cnt']

poly=PolynomialFeatures(degree=2, include_bias=False)
X_poly=poly.fit_transform(X)

print("Original and Polynomial Features : ")
print(pd.DataFrame(X_poly,columns=['temp','temp^2']).head())

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
X_poly_train,X_poly_test,Y_poly_train,Y_poly_test = train_test_split(X_poly,test_size=0.2,random_state=42)

# Train and evaluate model with original features
model_original=LinearRegression()
model_original.fit(X_train,Y_train)

Y_pred_original=model_original.predict(X_test)
mse_original=mean_squared_error(Y_test,Y_pred_original)

# Train and evaluate model with Polynomial features
model_poly=LinearRegression()
model_poly.fit(X_poly_train,Y_train)

Y_pred_poly=model_poly.predict(X_poly_test)

mse_poly=mean_squared_error(Y_test,Y_pred_poly)

print(f"MSE (Original) : {mse_original:.2f}")
print(f"MSE (Polynomial) : {mse_poly:.2f}")