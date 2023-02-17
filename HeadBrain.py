import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


Dataset = pd.read_csv('HeadBrain.csv')

print("Shape :",Dataset.shape)

X = Dataset['Head Size(cm^3)'].values
Y = Dataset['Brain Weight(grams)'].values

print("before reshape \:" ,X)

X = X.reshape((237,1))  #or ((-1,1))
print("after reshape \:" ,X)


reg = LinearRegression()
reg = reg.fit(X,Y)

y_pred = reg.predict(X)

r2 = reg.score(X,Y)

print(r2)