import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)

df=df[['Pclass','Sex','Age','Fare','Embarked','Survived']]

df.fillna({'Age':df['Age'].median()},inplace=True)
df.fillna({'Embarked':df['Embarked'].mode()},inplace=True)

X=df.drop(columns=['Survived'])
Y=df['Survived']

pre_processor=ColumnTransformer(
    transformers= [
        ('num', StandardScaler(),['Age','Fare']),
        ('cat', OneHotEncoder(),['Pclass','Sex','Embarked'])
    ]
)

X_preprocessed=pre_processor.fit_transform(X)

log_model=LogisticRegression()
loq_scores=cross_val_score(log_model,X_preprocessed,Y,cv=5,scoring='accuracy')

print(f"Logistic Regression Accuracy :{loq_scores.mean():.2f}")

rf_model=RandomForestClassifier(random_state=42)
rf_scores=cross_val_score(rf_model,X_preprocessed,Y,cv=5,scoring='accuracy')

print(f"Random Forrest Accuracy :{rf_scores.mean():.2f}")

param_grid={
    'n_estimators':[50,100,200],
    'max_depth':[None,10,20],
    'min_samples_split':[2,5,10]
}

grid_search=GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_preprocessed,Y)

print("Best Parameters : ",grid_search.best_params_)
print("Best Accuracy : ",grid_search.best_score_)