
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = pd.read_csv('mail_data.csv')
print(df)
data =df.where(pd.notnull(df),'')
print(data)
data.info()
data.shape
data.loc[data['Category'] == 'ham', 'Category'] = 1
data.loc[data['Category'] == 'spam', 'Category'] = 0
x = data['Message']
y = data['Category']
# PRINTING THE VALUE OF x AND y
print("PRINTING THE VALUE OF x")
print(x)
# printing the value of Y
print("printing the value of Y")
print(y)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state= 3 )
x.shape
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

feature_extraction = TfidfVectorizer(min_df = 1, stop_words= 'english', lowercase=True)

x_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression()
model.fit(x_train_features, Y_train)
predTrain = model.predict(x_train_features)
accuracyTrain = accuracy_score(Y_train, predTrain)
print('Accuract of the Training Model:', accuracyTrain)

predTest = model.predict(X_test_features)
accuracyTest = accuracy_score(Y_test, predTest)
print('Accuracy of the Test Set:', accuracyTest)

input = ["Congratulations! You have been selected as the winner of our monthly prize draw. You've won a brand new iPhone 12 Pro Max!To claim your prize, simply click on the link below and provide your shipping details. Hurry, as this offer is only valid for the next 24 hours!Please note that failure to claim your prize within the specified time frame will result in forfeiture.Thank you for participating in our contest, and we look forward to hearing from you soon!"]
input_features = feature_extraction.transform(input)

predInput = model.predict(input_features)
print(predInput)

print(type(predInput))

if(predInput[0] == 1):
  print("Its a Ham")
else:
  print("Its a Spam")

