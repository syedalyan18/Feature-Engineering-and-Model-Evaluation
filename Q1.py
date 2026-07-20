import pandas as pd
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)

print("Dataset Info :\n", df.info())
print("Dataset Preview :\n", df.head())
categorical_features=df.select_dtypes(include=["object"]).columns
numerical_features=df.select_dtypes(include=["int64","float64"]).columns

print("Categorical Features : \n",categorical_features.tolist())
print("Numerical Features : \n",numerical_features.tolist())

print("Categorical Features Summary: \n")
for col in categorical_features:
    print(f"{col}:\n",df[col].value_counts())

print("Numerical Features Summary: \n")
print(df[numerical_features].describe())