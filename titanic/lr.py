import pandas as pd 
import numpy as np
import os

df = pd.read_csv('datasets/train.csv', index_col='PassengerId')

# Remove passengersId, Name, Ticket and Cabin
titanic = df.reset_index()
removed_col = ['PassengerId', 'Name', 'Cabin', 'Ticket']
titanic = titanic.drop(removed_col, axis=1)

# fill Age with median
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

# fill Embarked with mode
embarked_mode = titanic['Embarked'].mode()[0]
titanic['Embarked'] = titanic['Embarked'].fillna(embarked_mode)



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

X = titanic.drop(['Survived'], axis=1)
y = titanic['Survived']

# Label econde sex col
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])


# Onehot encode Embarked col
ohe = OneHotEncoder(drop='first', sparse_output=False)
embarked_encoded = ohe.fit_transform(X[['Embarked']])
embarked_df = pd.DataFrame(embarked_encoded, columns=ohe.get_feature_names_out(['Embarked']))
X = pd.concat([X.drop(['Embarked'], axis=1).reset_index(drop=True), embarked_df.reset_index(drop=True)], axis=1)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#split into test train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Model 
model = LogisticRegression(C=0.5, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy Score", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



## kaggle test submission
test_df = pd.read_csv('./datasets/test.csv')
test = test_df.drop(removed_col, axis=1)
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0]) 
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
test

test['Sex'] = le.fit_transform(test['Sex'])

ohe = OneHotEncoder(drop='first', sparse_output=False)

embarked_test_df = ohe.fit_transform(test[['Embarked']])
embarked_df = pd.DataFrame(embarked_test_df, columns=ohe.get_feature_names_out(['Embarked']))

final = pd.concat([test.drop(['Embarked'], axis=1).reset_index(drop=True), embarked_df.reset_index(drop=True)], axis=1)

test_pred = model.predict(final)

os.makedirs('./submissions', exist_ok=True)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': test_pred})
submission.to_csv('./submissions/lr.csv', index=False)
