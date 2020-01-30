import pandas as pd
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

## model pipelines
import models.random_forest as rf

train = pd.read_csv("data/train.csv")
train.set_index('Accident_ID',inplace=True)
test = pd.read_csv("data/test.csv")

target = "Severity"
X = train.drop(columns=[target])
y = train[target]

# Train
X_train, X_test, y_train, y_test  = train_test_split(X,y,stratify=y,test_size=0.1,random_state=5)
model = rf.train(X_train,y_train)

# average="weighted"
y_pred = model.predict(X_test)
print("Cross validation score".center(25,"-"))
print(rf.f1_score(y_test,y_pred,average="weighted"))

# Submission - test.csv
output = model.predict(test.drop(columns=['Accident_ID']))
result = pd.DataFrame({
    "Accident_ID": test['Accident_ID'],
    "Severity": output
})
out_path = './data/random/Result.csv'
result.to_csv(out_path,index=False)
