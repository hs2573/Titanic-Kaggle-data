# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential


def assign_sex(row):
	if row['Sex'] == 'male':
		return 1
	else:
		return 0



train_data = pd.read_csv("titanic_train.csv", sep=",")



train_data = train_data.drop(["PassengerId", "Name", "Cabin", "Embarked", "Ticket", "Fare"], axis=1)


train_data['Sex'] = train_data.apply(assign_sex, axis=1)

train_data.fillna(0, inplace=True)

X = np.array(train_data.iloc[:, 1:])
y = np.ravel(train_data.Survived)

print X[0,:]



#model for a binary classification to predict wine type
model = Sequential()
model.add(Dense(24, activation='relu', input_shape=(5,)))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
			   optimizer='adam',
			   metrics=['accuracy'])
model.fit(X, y, epochs=500, batch_size=1, verbose=1)


model.save("titanic.model")

import numpy as np 
import pandas as pd 
from keras.models import load_model, Sequential
from keras.layers import Dense 



def assign_sex(row):
	if row['Sex'] == 'male':
		return 1
	else:
		return 0


test_data = pd.read_csv("titanic_test.csv", sep=",")



test_data = test_data.drop(["PassengerId", "Name", "Cabin", "Embarked", "Ticket", "Fare"], axis=1)


test_data['Sex'] = test_data.apply(assign_sex, axis=1)

test_data.fillna(0, inplace=True)

X_test = np.array(test_data)




model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(5,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
			   optimizer='adam',
			   metrics=['accuracy'])


model = load_model("titanic.model")

predictions = np.round(model.predict(X_test))
predictions1 = model.predict(X_test)
predictions = np.array(predictions, dtype=np.int64)

print predictions.shape
