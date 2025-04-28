import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#Load dataset
df = pd.read_csv('Heart Prediction Quantum Dataset.csv')
X = df.drop('HeartDisease', axis=1)
y = np.array(df[['HeartDisease']]).flatten()

#Print dataset shapes
print("X shape:", X.shape)
print("y shape:", y.shape)

#Split the dataset into training and testing;  80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    

#Scale our features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Define different kernels for SVC
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
classifiers = {}
accuracies = {}

#Begin SVC
print('\nBegin SVC:')

#Train and evaluate SVM models with different kernels
print("Different Kernel results:")
for kernel in kernels:
    clf = SVC(kernel=kernel, C=1.0)
    clf.fit(X_train, y_train)
    classifiers[kernel] = clf
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[kernel] = accuracy
    print(f"Test Accuracy with {kernel} kernel: {accuracy:.2f}")

# Find the kernel with the highest accuracy
best_kernel = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_kernel]
print(f"\nBest kernel: {best_kernel} with accuracy: {best_accuracy:.2f}")

# Train the final SVC model with the best kernel
clf = SVC(kernel=best_kernel, C=1.0)
clf.fit(X_train, y_train)

print(f'SVC Model Score: {clf.score(X_train, y_train)}')
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'SVC Accuracy score: {accuracy}\n')
confusionMatrix = confusion_matrix(y_train, clf.predict(X_train))
print(confusionMatrix)

print(clf.predict(X_test))

print(classification_report(y_test, clf.predict(X_test)))
lbl = ['Not present', 'Present']
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=lbl)
disp.plot()
plt.title('SVC - Heart Disease')

#Begin logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

#Y-intercept and coefs
print('\nBegin Logistic Regression: ')
print(f'Y - intercept: {model.intercept_}')
print(f'Coefs: {model.coef_}\n')

#Logistic Regression scores
print(f'Logistic Regression Model Score: {model.score(X_train, y_train)}')
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy score: {accuracy}\n')
confusionMatrix = confusion_matrix(y_train, model.predict(X_train))
print(confusionMatrix)

print(model.predict(X_test))

print(classification_report(y_test, model.predict(X_test)))
lbl = ['Not present', 'Present']
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=lbl)
disp.plot()
plt.title('Logistic Regression - Heart Disease')
plt.show()