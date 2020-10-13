#Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

#Read the dataset
dataset = pd.read_csv(r'data/wine.csv')
print(dataset.head(5))

#NA values in the dataset
print("Count of NA values:")
print(dataset.isna().sum())

#Filter independent and dependent variables
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, [0]].values

#Split data into training and testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify = y)

#Preprocess the dataset
scaler = StandardScaler()
#scaler = Normalizer()
#scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train a classifier
classifier = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')
classifier.fit(X_train, y_train.ravel())

#Predcitions
y_pred = classifier.predict(X_test)

#Results
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))

# =============================================================================
# from matplotlib import pyplot as plt
# plt.scatter(X[:50, 2], X[:50, 3])
# plt.scatter(X[50:100, 2], X[50:100, 3])
# plt.scatter(X[100:150, 2], X[100:150, 3])
# plt.show()
# =============================================================================
