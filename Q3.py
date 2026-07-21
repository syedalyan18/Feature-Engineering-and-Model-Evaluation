import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)

print("Dataset Info :\n", df.info())
print("Dataset Preview :\n", df.head())

df_one_hot=pd.get_dummies(df,columns=['Sex', 'Embarked'],drop_first=True)

print("One-Hot Encoded Dataset : \n")
print(df_one_hot.head())

label_encoder=LabelEncoder()
df['Pclass_encoded']=label_encoder.fit_transform(df['Pclass'])

print("Label Encoded Dataset : \n")
print(df[['Pclass','Pclass_encoded']].head())

df['Ticket_frequency']=df['Ticket'].map(df['Ticket'].value_counts())

print("\n Frequency Encoded Feature :")
print(df[['Ticket','Ticket_frequency']].head())

x=df_one_hot.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
y=df['Survived']

X=x.dropna()
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=200)
model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)

print("Accuracy with One-Hot encoding : ",accuracy_score(Y_test,Y_pred))