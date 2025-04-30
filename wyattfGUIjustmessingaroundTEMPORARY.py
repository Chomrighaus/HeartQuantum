import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ALGORITHMS ////////////////////////////////////////////////////////////
df = pd.read_csv('Heart Prediction Quantum Dataset.csv')
X = df.drop('HeartDisease', axis=1)
y = np.array(df[['HeartDisease']]).flatten()

#Determine feature ranges for sliders
feature_ranges = {col: (float(df[col].min()), float(df[col].max()))
                  for col in X.columns if col.lower() != 'gender'}

#Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifiers = {}
accuracies = {}

#Begin K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
classifiers['K-Nearest Neighbors'] = knn
y_pred = knn.predict(X_test)
accuracies['K-Nearest Neighbors'] = accuracy_score(y_test, y_pred)

#Begin Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
classifiers['Random Forest'] = rf
y_pred = rf.predict(X_test)
accuracies['Random Forest'] = accuracy_score(y_test, y_pred)

#Begin Naive Bayes
nb = GaussianNB()
bn = nb
bn.fit(X_train, y_train)
classifiers['Naive Bayes'] = nb
y_pred = nb.predict(X_test)
accuracies['Naive Bayes'] = accuracy_score(y_test, y_pred)

#Begin SVC
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svm_clfs, svm_acc = {}, {}
for kernel in kernels:
    clf = SVC(kernel=kernel, C=1.0, probability=True)
    clf.fit(X_train, y_train)
    svm_clfs[kernel] = clf
    svm_acc[kernel] = accuracy_score(y_test, clf.predict(X_test))
best_kernel = max(svm_acc, key=svm_acc.get)
best_svc = svm_clfs[best_kernel]
best_svc.fit(X_train, y_train)
classifiers[f'SVC ({best_kernel})'] = best_svc
accuracies[f'SVC ({best_kernel})'] = svm_acc[best_kernel]

#Begin Logistic Regression
log = LogisticRegression()
log.fit(X_train, y_train)
classifiers['Logistic Regression'] = log
y_pred = log.predict(X_test)
accuracies['Logistic Regression'] = accuracy_score(y_test, y_pred)

feature_names = list(X.columns)

# GUI /////////////////////////////////////////////////////////////////////
class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Heart Disease Predictor')
        self.geometry('800x600')
        container = tk.Frame(self)
        container.pack(fill='both', expand=True)
        self.frames = {}
        for F in (MainMenu, DataEntry, GraphDisplay):
            page = F(parent=container, controller=self)
            self.frames[F] = page
            page.grid(row=0, column=0, sticky='nsew')
        self.show_frame(MainMenu)

    def show_frame(self, cls):
        self.frames[cls].tkraise()

class MainMenu(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        ttk.Label(self, text='Main Menu', font=('Arial',24)).pack(pady=20)
        ttk.Button(self, text='Enter Patient Data',
                   command=lambda: controller.show_frame(DataEntry)).pack(pady=10)
        ttk.Button(self, text='View Model Comparison',
                   command=lambda: controller.show_frame(GraphDisplay)).pack(pady=10)

class DataEntry(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        control = tk.Frame(self)
        control.pack(side='left', fill='y', padx=10, pady=10)
        plot = tk.Frame(self)
        plot.pack(side='right', fill='both', expand=True, padx=10, pady=10)

        ttk.Label(control, text='Enter Patient Data', font=('Arial',18)).pack(pady=5)
        self.sliders = {}
        for feat in feature_names:
            if feat.lower() != 'gender':
                mn, mx = feature_ranges[feat]
                resolution = 0.1 if 'quantum' in feat.lower() else 1.0
                ttk.Label(control, text=feat).pack(anchor='w')
                s = tk.Scale(control,
                             from_=mn,
                             to=mx,
                             resolution=resolution,
                             orient='horizontal',
                             length=200,
                             showvalue=True,
                             command=lambda v: self.update_plot())
                s.set(mn)
                s.pack(pady=2)
                self.sliders[feat] = s
        ttk.Label(control, text='Gender').pack(anchor='w', pady=(10,0))
        self.gender_box = ttk.Combobox(control, values=['Male','Female'], state='readonly')
        self.gender_box.current(0)
        self.gender_box.pack(pady=2)
        self.gender_box.bind('<<ComboboxSelected>>', lambda e: self.update_plot())

        ttk.Label(control, text='Algorithm').pack(pady=(10,0))
        self.algo_box = ttk.Combobox(control, values=list(classifiers.keys()), state='readonly')
        self.algo_box.current(0)
        self.algo_box.pack(pady=2)
        self.algo_box.bind('<<ComboboxSelected>>', lambda e: self.update_plot())

        ttk.Button(control, text='Submit', command=self.submit).pack(pady=10)
        ttk.Button(control, text='Back',
                   command=lambda: controller.show_frame(MainMenu)).pack()

        self.fig = Figure(figsize=(4,4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.update_plot()

    def get_sample(self):
        vals = []
        for feat in feature_names:
            if feat.lower() == 'gender':
                vals.append(1 if self.gender_box.get()=='Male' else 0)
            else:
                vals.append(float(self.sliders[feat].get()))
        return np.array(vals)

    def update_plot(self):
        self.ax.clear()
        model = classifiers[self.algo_box.get()]
        probs = model.predict_proba(X_test)[:,1]
        idx = np.arange(len(probs))
        #Color-coded test samples by true label
        idx0 = idx[y_test == 0]
        idx1 = idx[y_test == 1]
        self.ax.scatter(idx0, probs[y_test == 0], color='blue', alpha=0.5, label='No Disease')
        self.ax.scatter(idx1, probs[y_test == 1], color='red', alpha=0.5, label='Disease')
        sample = self.get_sample()
        scaled = scaler.transform([sample])
        sp = model.predict_proba(scaled)[0][1]
        self.ax.scatter([0], [sp], color='orange', s=100, edgecolors='black', label='Your Patient')
        self.ax.set_ylim(0,1)
        self.ax.set_xlabel('Test Sample Index')
        self.ax.set_ylabel('Pred Prob of Disease')
        self.ax.set_title(self.algo_box.get())
        self.ax.legend()
        self.canvas.draw()

    def submit(self):
        sample = self.get_sample()
        model = classifiers[self.algo_box.get()]
        pred = model.predict(scaler.transform([sample]))[0]
        proba = model.predict_proba(scaler.transform([sample]))[0]
        msg = (f"{'Disease Present' if pred else 'No Disease'}\n\n"
               + f"No Disease: {proba[0]:.0%}\nDisease: {proba[1]:.0%}")
        messagebox.showinfo('Result', msg)

class GraphDisplay(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        ttk.Label(self, text='Model Comparison', font=('Arial',18)).pack(pady=10)
        ttk.Button(self, text='Show Test Accuracies', command=self.show_acc).pack(pady=5)
        ttk.Button(self, text='Back', command=lambda: controller.show_frame(MainMenu)).pack(pady=10)

    def show_acc(self):
        plt.figure()
        names, scores = list(accuracies.keys()), list(accuracies.values())
        bars = plt.bar(names, scores, width=0.5)
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{score:.2f}',
                     ha='center', va='bottom')
        plt.ylim(0,1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    app = MainApp()
    app.mainloop()
