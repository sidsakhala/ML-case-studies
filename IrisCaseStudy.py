import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def MarvellousKNeighborsClassifier():
    # df = pd.read_csv("iris.csv")
    Dataset = load_iris()       
    # print(Dataset)
    Data = Dataset.data
    Target = Dataset.target

    print(Data)
    print("=======================")
    print(Target)
    Dataset['Species'].value_counts()
    Data_train, Data_test,Target_train,Target_test = train_test_split(Data, Target,test_size = 0.5)

    Classifier = KNeighborsClassifier()

    Classifier.fit(Data_train,Target_train)

    Prediction = Classifier.predict(Data_test)

    Accuracy = accuracy_score(Target_test,Prediction)

    return Accuracy

def main():
    Ret = MarvellousKNeighborsClassifier()

    print("Accuracy of Iris dataset with KNN is ",Ret *100)

if __name__ == '__main__':
    main()