import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, d2_tweedie_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#load dataset
df = pd.read_csv('Heart Prediction Quantum Dataset.csv')
X = df.drop('HeartDisease', axis=1)
y = np.array(df[['HeartDisease']]).flatten()

#print dataset shapes to make sure we have all data accounted for
print("X shape:", X.shape)
print("y shape:", y.shape)

#split data and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#define classifiers
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100)
nb = GaussianNB()
classifiers = {}
accuracies = {}

###################################################################################
###################################################################################
#Begin KNN
print('\nBegin K-Nearest Neighbors:')
knn.fit(X_train, y_train)
classifiers['knn'] = knn
print(f"KNN Model Score: {knn.score(X_train, y_train)}")
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
accuracies['knn'] = acc
print(f"KNN Accuracy score: {acc:.2f}\n")
confusionMatrix = confusion_matrix(y_train, knn.predict(X_train))
print(confusionMatrix)
print(knn.predict(X_test))
print(classification_report(y_test, knn.predict(X_test)))
lbl = ['Not present', 'Present']
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=lbl)
disp.plot()
plt.title('KNN - Heart Disease')

####################################################################################
#Begin Random Forest
print('\nBegin Random Forest:')
rf.fit(X_train, y_train)
classifiers['rf'] = rf
print(f"Random Forest Model Score: {rf.score(X_train, y_train)}")
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
accuracies['rf'] = acc
print(f"Random Forest Accuracy score: {acc:.2f}\n")
confusionMatrix = confusion_matrix(y_train, rf.predict(X_train))
print(confusionMatrix)
print(rf.predict(X_test))
print(classification_report(y_test, rf.predict(X_test)))

disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=lbl)
disp.plot()
plt.title('Random Forest - Heart Disease')

#####################################################################################
#Begin Naive Bayes
print('\nBegin Naive Bayes:')
nb.fit(X_train, y_train)
classifiers['nb'] = nb
print(f"Naive Bayes Model Score: {nb.score(X_train, y_train)}")
y_pred = nb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
accuracies['nb'] = acc
print(f"Naive Bayes Accuracy score: {acc:.2f}\n")
confusionMatrix = confusion_matrix(y_train, nb.predict(X_train))
print(confusionMatrix)
print(nb.predict(X_test))
print(classification_report(y_test, nb.predict(X_test)))

disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=lbl)
disp.plot()
plt.title('Naive Bayes - Heart Disease')

###################################################################################
###################################################################################
# Find the classifier with the highest accuracy
best = max(accuracies, key=accuracies.get)
print(f"\nBest classifier: {best} with accuracy: {accuracies[best]:.2f}\n")

plt.show()

#Optional: cross-validation (uncomment if needed) Could be useful for showing how good/bad our model performs in general, not just in one biased split of data potentially
#X_all = StandardScaler().fit_transform(X)             ##X_all is full feature matrix without train
#kf = KFold(n_splits=5, shuffle=True)                  ##Split X_all into 5 parts of (roughly) equal size, then shuffle that shi
#for name, clf in classifiers.items():                 ##Runs five independent rounds of torurous interrigations, then returns 5 accuracy scores, takes the mean of them, gets the StandDevi to see-
#    scores = cross_val_score(clf, X_all, y, cv=kf)     #-how stable the model was accross the different splits. High mean = good performance. small std = consistent model. high std = overfitting or underfitting issues
#    print(f"{name} 5-fold CV Accuracy: {scores.mean():.2f} | std: (+/-) {scores.std():.2f}")
