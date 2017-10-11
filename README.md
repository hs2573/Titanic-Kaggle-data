# Titanic-Kaggle-data
The data from Kaggle <https://www.kaggle.com/c/titanic>
The program using the ANN model to train the data.
First download the data from the Kaggle <https://www.kaggle.com/c/titanic/data>. It has three .csv files which are test.csv, train.csv and gender_submission.csv. Use the “Spyder” to program this project. Import the Numpy, Pandas, Matplotlib, tensorflow and Keras. Read the train.csv them it could show the data
Then we drop some data which were not used, such as "PassengerId", "Name", "Ticket", "Cabin" and "Fare", then change the words to number.  Then create the ANN module one input, output layer and three hidden layers. Set up the epochs = 1000. Then we could get the loss and accuracy. This got 88% accuracy.
