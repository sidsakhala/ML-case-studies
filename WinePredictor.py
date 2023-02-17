import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def MarvellousKNeighborsClassifier():
    dataset = pd.read_csv("WinePredictor.csv")
    
    Data = dataset.drop(['Class'], axis  = 1)
    Target = dataset['Class']

    print(Data)
    print("=======================")
    print(Target)
    
    Data_train, Data_test,Target_train,Target_test = train_test_split(Data, Target,test_size = 0.3)

    Classifier = KNeighborsClassifier(n_neighbors = 3)

    Classifier.fit(Data_train,Target_train)

    # Prediction = Classifier.predict([[10,2,2,20,130,3,2,0.22,3,5,1,4,1000]])
    Prediction = Classifier.predict([[12.08,1.83,2.32,18.5,81,1.6,1.5,0.52,1.64,2.4,1.08,2.27,480]])


    # Accuracy = accuracy_score(Target_test,Prediction)

    print("class : ",Prediction)

    # return Accuracy

def main():
    Ret = MarvellousKNeighborsClassifier()

    # print("Accuracy of Wine dataset with KNN is ",Ret *100)

if __name__ == '__main__':
    main()