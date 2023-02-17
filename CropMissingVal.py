import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm


# Dataset = pd.read_csv('Crop_recommendation.csv')

Dataset = pd.read_csv('MissingValue.csv')


print("Shape :",Dataset.shape)

X = Dataset.drop(['K','temperature','humidity','ph','rainfall','label'],axis = 1)
# X = Dataset.drop([Longitude,Latitude,pH,EC,OC,N,P,K,S,Z ,Fe,Cu,M,B],axis = 1)
# X = Dataset.drop(['Longitude','Latitude','K'],axis = 1)
# X = Dataset.drop(columns='Penjualan (pcs)')
Y = Dataset['K']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=4)

print(X.shape)
print(Y.shape)

reg = LinearRegression()
# reg = svm.SVR()
reg = reg.fit(X,Y)

y_pred = reg.predict([[90,42]])

r2 = reg.score(X,Y)

# print(reg.coef_)
# print(reg.intercept_)
print("predicted :",y_pred)
print("score is :",r2*100)


# from sklearn.ensemble import GradientBoostingClassifier
# grad = GradientBoostingClassifier().fit(X_train, y_train)