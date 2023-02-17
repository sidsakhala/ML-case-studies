import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB


Dataset = pd.read_csv('PlayPredictor.csv')
# Feature = Dataset.drop('Play',axis=1)
# Target = Dataset['Play']

w = Dataset.Whether
T = Dataset.Temperature
play = Dataset.Play

le  = preprocessing.LabelEncoder()

w_encoded = le.fit_transform(w)
t_encoded = le.fit_transform(T)
label = le.fit_transform(play)

features  = list(zip(w_encoded, t_encoded))

# Data_Train,Data_Test,Target_Train,Target_Test = train_test_split(Data,Target,test_size = 0.3)

# classifier = KNeighborsClassifier(n_neighbors =3)
classifier = GaussianNB()

classifier.fit(features, label)

prediction = classifier.predict([[1,1]])  #0:Overcast, 2:Mild 

# accuracy = accuracy_score(Target_Test,prediction)
print(prediction)