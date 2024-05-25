#Parameter for KNN is age of the abalone.
#Business question: in culinary, abalones can be served either as an appetizer or an entree, depending on its size.
#Abalones above younger than 9 years old can be served as appetizers, while the rest tend to be served as an entree.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline          
#importing libraries

#READING AND PREPARING DATA

#creating list with column names
column_names = ['Sex','Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight','Rings']

#reading dataset
df = pd.read_csv('C:/Users/Ricardo/Downloads/abalone/abalone.data.csv',
                 header=None, names=['data'])

#no headers, first row will be column_names, names = [data] will force the dataset to have only one column

df = df['data'].str.split(',', expand=True)          #separate the dataframe according to the commas
df.columns = column_names
df.to_csv('separated_data.csv', index=False)

#convert numbers from string to float

columns_to_convert = ['Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight','Rings']
df[columns_to_convert] = df[columns_to_convert].astype(float)

#create age column: ring column + 1.5

df['age'] = df['Rings'] + 1.5

#converting the different utilities of abalone to numbers. 
#0 are abalones that will be user for appetizer, 1 can be used for entree

#VISUALIZING

sns.pairplot(df, hue = 'maturity')
#hue is the parameter we are using, therefore, the data will be seaprated according to maturity

#SPLITTING DATAFRAME

from sklearn.model_selection import train_test_split
X = df.drop('Sex', axis = 1)
X.drop('maturity', axis = 1)
Y = df['maturity']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=12)

#SCALING

from sklearn.preprocessing import MinMaxScaler    
#MinMaxScaler standardizes values from 0 to 1, using as parameters the lowest and the highest values

df['maturity'] = df['age'].apply(lambda x: 0 if x < 9 else 1)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

scaling_X = scaler.fit_transform(X)

#USING K = 9

from sklearn.neighbors import KNeighborsClassifier
knn_project = KNeighborsClassifier(n_neighbors = 9)
knn_project.fit(X_train, Y_train)

y_predict = knn_project.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, y_predict)
print('Accuracy:', accuracy)

k_values = [i for i in range (1,200)]
scores = []

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import cross_val_score
for k in k_values:
    knn_project = KNeighborsClassifier(n_neighbors = k)
    score = cross_val_score(knn_project, X, Y, cv = 5)
    scores.append(np.mean(score))

sns.lineplot(x = k_values, y = scores, marker = '.')
plt.xlabel('K Values')
plt.ylabel('Accuracy Score')
