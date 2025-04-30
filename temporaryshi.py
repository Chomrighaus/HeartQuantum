import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ---------------------- ALGORITHM SETUP ----------------------
df = pd.read_csv('Heart Prediction Quantum Dataset.csv')
X = df.drop('HeartDisease', axis=1)
y = np.array(df[['HeartDisease']]).flatten()

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifiers = {}
accuracies = {}

# Begin K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
print("\nBegin K-Nearest Neighbors:")
knn.fit(X_train, y_train)
classifiers['K-Nearest Neighbors'] = knn
e_val = knn.score(X_train, y_train)
print(f"KNN Model Score: {e_val}")
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
accuracies['K-Nearest Neighbors'] = acc
print(f"KNN Accuracy score: {acc:.2f}")

# Begin Random Forest
rf = RandomForestClassifier(n_estimators=100)
print("\nBegin Random Forest:")
rf.fit(X_train, y_train)
classifiers['Random Forest'] = rf
e_val = rf.score(X_train, y_train)
print(f"Random Forest Model Score: {e_val}")
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
accuracies['Random Forest'] = acc
print(f"Random Forest Accuracy score: {acc:.2f}")

# Begin Naive Bayes
nb = GaussianNB()
print("\nBegin Naive Bayes:")
nb.fit(X_train, y_train)
classifiers['Naive Bayes'] = nb
e_val = nb.score(X_train, y_train)
print(f"Naive Bayes Model Score: {e_val}")
y_pred = nb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
accuracies['Naive Bayes'] = acc
print(f"Naive Bayes Accuracy score: {acc:.2f}")

# Begin SVC (from Irie)
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svm_clfs = {}
svm_acc = {}
print("\nBegin SVC:")
print("Different Kernel results:")
for kernel in kernels:
    clf = SVC(kernel=kernel, C=1.0, probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    a = accuracy_score(y_test, y_pred)
    svm_clfs[kernel] = clf
    svm_acc[kernel] = a
    print(f"Test Accuracy with {kernel} kernel: {a:.2f}")
best_kernel = max(svm_acc, key=svm_acc.get)
best_svc = svm_clfs[best_kernel]
print(f"\nBest kernel: {best_kernel} with accuracy: {svm_acc[best_kernel]:.2f}")
best_svc.fit(X_train, y_train)
classifiers[f"SVC ({best_kernel})"] = best_svc
accuracies[f"SVC ({best_kernel})"] = svm_acc[best_kernel]

# Begin Logistic Regression (from Irie)
log = LogisticRegression()
print("\nBegin Logistic Regression:")
log.fit(X_train, y_train)
print(f"Y - intercept: {log.intercept_}")
print(f"Coefs: {log.coef_}\n")
classifiers['Logistic Regression'] = log
y_pred = log.predict(X_test)
acc = accuracy_score(y_test, y_pred)
accuracies['Logistic Regression'] = acc
print(f"Logistic Regression Accuracy score: {acc:.2f}")

# Feature list for GUI fields
feature_names = list(X.columns)

# ---------------------- GUI SETUP ----------------------
class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Heart Disease Predictor")
        self.geometry("500x650")
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)
        self.frames = {}
        for F in (MainMenu, DataEntry, GraphDisplay):
            page = F(parent=container, controller=self)
            self.frames[F] = page
            page.grid(row=0, column=0, sticky="nsew")
        self.show_frame(MainMenu)

    def show_frame(self, cls):
        self.frames[cls].tkraise()

class MainMenu(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        ttk.Label(self, text="Main Menu", font=("Arial", 24)).pack(pady=20)
        ttk.Button(self, text="Enter Patient Data", 
                   command=lambda: controller.show_frame(DataEntry)).pack(pady=10)
        ttk.Button(self, text="View Model Comparison", 
                   command=lambda: controller.show_frame(GraphDisplay)).pack(pady=10)

class DataEntry(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        ttk.Label(self, text="Data Entry", font=("Arial",20)).pack(pady=10)
        self.entries = {}
        for feat in feature_names:
            ttk.Label(self, text=feat).pack(anchor='w', padx=20)
            ent = ttk.Entry(self)
            ent.pack(fill='x', padx=20, pady=2)
            self.entries[feat] = ent
        self.algo_var = tk.StringVar()
        ttk.Label(self, text="Choose Algorithm").pack(pady=(10,0))
        self.algo_box = ttk.Combobox(self, textvariable=self.algo_var,
                                     values=list(classifiers.keys()), state="readonly")
        self.algo_box.current(0)
        self.algo_box.pack(pady=5)
        ttk.Button(self, text="Submit", command=self.submit_data).pack(pady=10)
        ttk.Button(self, text="Back to Menu", 
                   command=lambda: controller.show_frame(MainMenu)).pack()

    def submit_data(self):
        vals = []
        for feat in feature_names:
            v = self.entries[feat].get()
            if feat.lower() == "gender":
                if v.lower() in ["male", "m"]: num = 1
                elif v.lower() in ["female", "f"]: num = 0
                else:
                    messagebox.showerror("Input Error", "Enter 'male' or 'female'")
                    return
                vals.append(num)
            else:
                try:
                    vals.append(float(v))
                except:
                    messagebox.showerror("Input Error", f"Invalid value for {feat}")
                    return
        model = classifiers[self.algo_var.get()]
        pred = model.predict(scaler.transform([vals]))[0]
        proba = model.predict_proba(scaler.transform([vals]))[0]
        text = "Disease Present" if pred else "No Disease"
        info = (f"{text}\n\nConfidence:\n"
                f"No Disease: {proba[0]:.0%}\n"
                f"Disease: {proba[1]:.0%}")
        messagebox.showinfo("Prediction Result", info)

class GraphDisplay(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        ttk.Label(self, text="Model Comparison", font=("Arial",20)).pack(pady=10)
        ttk.Button(self, text="Show Test Accuracies", command=self.show_acc).pack(pady=5)
        # Scatter plot buttons for each algorithm
        for name in classifiers.keys():
            ttk.Button(self, text=f"Scatter - {name}", 
                       command=lambda n=name: self.show_scatter(n)).pack(pady=5)
        ttk.Button(self, text="Back to Menu", command=lambda: controller.show_frame(MainMenu)).pack(pady=10)

    def show_acc(self):
        names = list(accuracies.keys())
        scores = [accuracies[n] for n in names]
        plt.figure()
        bars = plt.bar(names, scores, width=0.5)
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{score:.2f}', ha='center', va='bottom')
        plt.ylim(0,1)
        plt.ylabel('Accuracy')
        plt.title('Test Set Accuracies')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def show_scatter(self, name):
        model = classifiers[name]
        # Predict probabilities if possible
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            # fallback to decision function
            probs = model.decision_function(X_test)
        plt.figure()
        idx0 = np.where(y_test == 0)[0]
        idx1 = np.where(y_test == 1)[0]
        plt.scatter(idx0, probs[idx0], color='blue', label='No Disease', alpha=0.7)
        plt.scatter(idx1, probs[idx1], color='red', label='Disease', alpha=0.7)
        plt.xlabel('Test Sample Index')
        plt.ylabel('Predicted Probability of Disease')
        plt.title(f'{name} Prediction Probabilities')
        plt.axhline(0.5, color='gray', linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
