import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def Tit():
    dataset = pd.read_csv("MarvellousTitanicDataset.csv")
    
    dataset.drop(["Sex","sibsp","Parch","Embarked"],axis = 1, inplace = True)

    Data = dataset.drop("Survived",axis  = 1)
    Target = dataset["Survived"]

    

    # print(Data)
    # print("=======================")
    # print(Target)
    
    # Data_train, Data_test,Target_train,Target_test = train_test_split(Data, Target,test_size = 0.3)

    Data_train, Data_test,Target_train,Target_test = train_test_split(Data,Target,test_size = 0.3,random_state = 101)


    Classifier = LogisticRegression()

    Classifier.fit(Data_train,Target_train)
    # Prediction = Classifier.predict([[10,2,2,20,130,3,2,0.22,3,5,1,4,1000]])
    # Prediction = Classifier.predict([[12.08,1.83,2.32,18.5,81,1.6,1.5,0.52,1.64,2.4,1.08,2.27,480]])

    Prediction = Classifier.predict(Data_test)
    
    Accuracy = accuracy_score(Target_test,Prediction)

    # print("class : ",Prediction)

    print("accuracy : ",Accuracy*100)

def main():
    Tit()

if __name__ == '__main__':
    main()