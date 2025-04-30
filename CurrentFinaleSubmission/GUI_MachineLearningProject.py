import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from models import run

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Heart Quantum: A Comparison Of Different Algorithms")
        self.state("zoomed")
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))

        container = tk.Frame(self)
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}

        for F in (MainMenu, GraphsWindow, AccuracyMatrixWindow, RandomForestWindow,
                  KNNWindow, NaiveBayesWindow, LogisticRegressionWindow, SVCLinearWindow,
                  SVCPolynomialKernalWindow, SVCRBFKernelWindow, SVCSigmoidKernelWindow,
                  AccuracyComparisonWindow, ScatterWindow,  EndPresentationWindow): # DataEntryWindow before EndPresentation Window
            frame = F(parent=container, controller=self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(MainMenu)

    def show_frame(self, page):
        frame = self.frames[page]
        frame.tkraise()


class MainMenu(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Heart Quantum: A Comparison Of Different Algorithms", font=("Arial", 18))
        label.pack(pady=10)
        sub_label = tk.Label(self, text="Wyatt, Irie, and Conner's Project!", font=("Arial", 14))
        sub_label.pack(pady=5)

        pages = [
                    ("Visible Confusion Matrix!", GraphsWindow),
                    ("Accuracy and Confusion Matrix!", AccuracyMatrixWindow),
                    ("Random Forest", RandomForestWindow),
                    ("KNN", KNNWindow),
                    ("Naive Bayes", NaiveBayesWindow),
                    ("Logistic Regression", LogisticRegressionWindow),
                    ("SVC (Linear Kernel)", SVCLinearWindow),
                    ("SVC (Polynomial Kernal)", SVCPolynomialKernalWindow),
                    ("SVC (RBF Kernal)", SVCRBFKernelWindow),
                    ("SVC (Sigmoid Kernal)", SVCSigmoidKernelWindow),
                    ("Accuracy Comparisons", AccuracyComparisonWindow),
                    ("Scatter Plots", ScatterWindow),
                    #("Data Entry", DataEntryWindow),
                    ("End of Presentation", EndPresentationWindow) 
            ]

        for text, page in pages:
            button = ttk.Button(self, text=text, command=lambda p=page: controller.show_frame(p))
            button.pack(pady=5)


class GraphsWindow(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Visible Confusion Matrix!", font=("Arial", 18))
        label.pack(pady=10)

        scroll_frame = ScrollableFrame(self)
        scroll_frame.pack(fill="both", expand=True)
        self.canvas_holder = scroll_frame.scrollable_frame

        btn = ttk.Button(self, text="Generate All Graphs", command=self.show_graphs)
        btn.pack(pady=10)

        ttk.Button(self, text="Back to Main Menu", 
                   command=lambda: controller.show_frame(MainMenu)).pack(pady=10, side="bottom")

    def show_graphs(self):
        for widget in self.canvas_holder.winfo_children():
            widget.destroy()
        result = run()
        for fig in result['figures']:
            canvas = FigureCanvasTkAgg(fig, master=self.canvas_holder)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=10)


class AccuracyMatrixWindow(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Accuracy and Confusion Matrix (Not Visible)!", font=("Arial", 18))
        label.pack(pady=10)

        scroll_frame = ScrollableFrame(self)
        scroll_frame.pack(fill="both", expand=True)
        self.text = tk.Text(scroll_frame.scrollable_frame, wrap="word", height=30, width=100)
        self.text.pack()

        ttk.Button(self, text="Load Results", command=self.display_info).pack(pady=10)
        ttk.Button(self, text="Back to Main Menu", 
                   command=lambda: controller.show_frame(MainMenu)).pack(pady=10, side="bottom")

    def display_info(self):
        self.text.delete("1.0", tk.END)
        result = run()
        for model_name, info in result['metrics'].items():
            self.text.insert(tk.END, f"{model_name}\nAccuracy: {info['accuracy']:.2f}\nConfusion Matrix:\n{info['conf_matrix']}\n\n")


class BaseDetailWindow(tk.Frame):
    def __init__(self, parent, controller, title, key):
        super().__init__(parent)
        label = tk.Label(self, text=title, font=("Arial", 18))
        label.pack(pady=10)

        scroll_frame = ScrollableFrame(self)
        scroll_frame.pack(fill="both", expand=True)
        self.text = tk.Text(scroll_frame.scrollable_frame, wrap="word", height=30, width=100)
        self.text.pack()

        ttk.Button(self, text="Show Details", command=lambda: self.display_info(key)).pack(pady=10)
        ttk.Button(self, text="Back to Main Menu", 
                   command=lambda: controller.show_frame(MainMenu)).pack(pady=10, side="bottom")

    def display_info(self, model_key):
        self.text.delete("1.0", tk.END)
        result = run()
        if model_key in result['metrics']:
            info = result['metrics'][model_key]
            self.text.insert(tk.END, f"Accuracy: {info['accuracy']:.2f}\n")
            self.text.insert(tk.END, f"Confusion Matrix:\n{info['conf_matrix']}\n")
            self.text.insert(tk.END, f"Report:\n{info['report']}\n")


class RandomForestWindow(BaseDetailWindow):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Random Forest", "Random Forest")

class KNNWindow(BaseDetailWindow):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "K-Nearest Neighbors", "KNN")

class NaiveBayesWindow(BaseDetailWindow):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Naive Bayes", "Naive Bayes")

class LogisticRegressionWindow(BaseDetailWindow):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Logistic Regression", "Logistic Regression")

class SVCLinearWindow(BaseDetailWindow):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Support Vector Classifier", "SVC (Linear Kernel)")

class SVCPolynomialKernalWindow(BaseDetailWindow):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Support Vector Classifier", "SVC (Polynomial Kernel)")

class SVCRBFKernelWindow(BaseDetailWindow):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Support Vector Classifier", "SVC (RBF Kernel)")

class SVCSigmoidKernelWindow(BaseDetailWindow):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "Support Vector Classifier", "SVC (Sigmoid Kernel)")

class AccuracyComparisonWindow(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Accuracy Comparisons", font=("Arial", 18))
        label.pack(pady=10)

        scroll_frame = ScrollableFrame(self)
        scroll_frame.pack(fill="both", expand=True)
        self.scroll_frame = scroll_frame.scrollable_frame

        ttk.Button(self, text="Compare Accuracies", command=self.plot_accuracies).pack(pady=10)
        ttk.Button(self, text="Back to Main Menu", 
                   command=lambda: controller.show_frame(MainMenu)).pack(pady=10, side="bottom")

    def plot_accuracies(self):
        result = run()
        model_names = list(result['metrics'].keys())
        accuracies = [m['accuracy'] for m in result['metrics'].values()]
        fig, ax = plt.subplots()
        ax.bar(model_names, accuracies)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy Comparison")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.scroll_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=10)

'''
class DataEntryWindow(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Data Entry", font=("Arial", 18))
        label.pack(pady=10)
        ttk.Label(self, text="Select Algorithm:").pack()
        self.algorithm_menu = ttk.Combobox(self, values=["Logistic Regression", "SVC", "KNN", "Random Forest"])
        self.algorithm_menu.pack(pady=5)
        ttk.Label(self, text="Enter Data (comma-separated):").pack()
        self.entry = ttk.Entry(self, width=50)
        self.entry.pack(pady=5)
        ttk.Button(self, text="Submit", command=self.submit).pack(pady=10)
        ttk.Button(self, text="Back to Main Menu", 
                   command=lambda: controller.show_frame(MainMenu)).pack(pady=10, side="bottom")

    def submit(self):
        algo = self.algorithm_menu.get()
        data = self.entry.get()
        print(f"Data submitted for {algo}: {data}")
'''

class ScatterWindow(tk.Frame):
    def __init__(self, parent, controller):
       super().__init__(parent)
       label = tk.Label(self, text="Scatter Plots", font=("Arial", 18))
       label.pack(pady=10)

       scroll_frame = ScrollableFrame(self)
       scroll_frame.pack(fill="both", expand=True)
       self.canvas_holder = scroll_frame.scrollable_frame

       btn = ttk.Button(self, text="Generate All Graphs", command=self.show_graphs)
       btn.pack(pady=10)

       ttk.Button(self, text="Back to Main Menu", 
                   command=lambda: controller.show_frame(MainMenu)).pack(pady=10, side="bottom")

    def show_graphs(self):
        for widget in self.canvas_holder.winfo_children():
            widget.destroy()
        result = run()
        for fig in result['scatter']:
            canvas = FigureCanvasTkAgg(fig, master=self.canvas_holder)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=10)


class EndPresentationWindow(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="End of Presentation", font=("Arial", 18))
        label.pack(pady=20)
        msg = tk.Label(self, text="Thank you, any questions? :D", font=("Arial", 14))
        msg.pack(pady=10)
        ttk.Button(self, text="Back to Main Menu", 
                   command=lambda: controller.show_frame(MainMenu)).pack(pady=10, side="bottom")