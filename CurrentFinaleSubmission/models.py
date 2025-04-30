import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA

def run():
    df = pd.read_csv('Heart Prediction Quantum Dataset.csv')
    X = df.drop('HeartDisease', axis=1)
    y = np.array(df['HeartDisease'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "SVC (Linear Kernel)": SVC(kernel='linear'),
        "SVC (Polynomial Kernel)": SVC(kernel='poly'),
        "SVC (RBF Kernel)": SVC(kernel='rbf'),
        "SVC (Sigmoid Kernel)": SVC(kernel='sigmoid'),
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB()
    }

    results = {
        "metrics": {},
        "figures": [],
        "scatter": []
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        results["metrics"][name] = {
            "accuracy": acc,
            "conf_matrix": cm,
            "report": report
        }

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Present", "Present"])
        disp.plot(ax=ax)
        ax.set_title(f"{name} - Confusion Matrix")
        results["figures"].append(fig)

    # Scatter Plots (using first two principal features for visualization)
    # PCA for 2D projection
    pca = PCA(n_components=2)
    X_test_2d = pca.fit_transform(X_test)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fig, ax = plt.subplots()
        scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_pred, cmap=plt.cm.RdYlGn, alpha=0.6, edgecolors='k')
        ax.set_title(f"{name} - Scatter Plot (PCA)")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        results["scatter"].append(fig)

    return results