# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:33:38 2024

@author: Lucas Friedrich

Score: 0.77511
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd 

train_file_path = "/../Users/DELL/Documents/Keegle/Titanic - Machine Learning from Disaster/train.csv"
test_file_path = "/../Users/DELL/Documents/Keegle/Titanic - Machine Learning from Disaster/test.csv"

train_data = pd.read_csv(train_file_path, delimiter=';')
test_data = pd.read_csv(test_file_path, delimiter=';')

test_data.head()
train_data.head()


women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of women who survived:", rate_women)
print("% of men who survived:", rate_men)

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")