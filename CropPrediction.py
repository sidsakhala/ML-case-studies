import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm

def classify():
    dataset = pd.read_csv("Crop_recommendation.csv")
    
    Data = dataset.drop(['temperature','humidity','ph','rainfall','label'], axis  = 1)
    Target = dataset['label']

    print(Data)
    print("=======================")
    print(Target)
    
    Data_train, Data_test,Target_train,Target_test = train_test_split(Data, Target,test_size = 0.3)

    # Classifier = KNeighborsClassifier(n_neighbors = 3)
    Classifier  = svm.SVC()

    Classifier.fit(Data_train,Target_train)

    # Prediction = Classifier.predict([[10,2,2,20,130,3,2,0.22,3,5,1,4,1000]])
    # Prediction = Classifier.predict([[14,37,15]])

    # Prediction = Classifier.predict([[810,14,70]])
    Prediction = Classifier.predict([[1105.24,13.84,79.22]])

    AccuracyPrediction = Classifier.predict(Data_test)

    Accuracy = accuracy_score(Target_test,AccuracyPrediction)

    print("crop : ",Prediction)
    print("accuracy : ",Accuracy)

    # return Accuracy

def main():
    Ret = classify()

    # print("Accuracy of Wine dataset with KNN is ",Ret *100)

if __name__ == '__main__':
    main()