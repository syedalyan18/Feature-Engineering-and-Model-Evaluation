import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

data=load_diabetes()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target

correlation_matrix=df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

correlated_features=correlation_matrix['target'].sort_values(ascending=False)
print("Features most correlated with target :")
print(correlated_features)

X=df.drop(columns=['target'])
Y=df['target']

mutual_info=mutual_info_regression(X,Y)

mi_df=pd.DataFrame({"Feature":X.columns,"Mutual Information":mutual_info})
mi_df=mi_df.sort_values(by="Mutual Information",ascending=False)

print("Mutual Information Scores :")
print(mi_df)

model=RandomForestRegressor(random_state=42)
model.fit(X,Y)

feature_importance=model.feature_importances_
importance_df=pd.DataFrame({"Feature":X.columns,"Importance":feature_importance})
importance_df=importance_df.sort_values(by="Importance",ascending=False)

print("Feature Importance from random Forest :")
print(importance_df)

plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'],importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance from Random Forest")
plt.show()