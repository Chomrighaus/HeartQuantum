import tkinter as tk
from tkinter import ttk

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Machine Learning Project GUI")
        self.geometry("400x300")

        # Container for all frames
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)

        self.frames = {}

        for F in (MainMenu, DataEntry, GraphDisplay):
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
        self.controller = controller

        label = ttk.Label(self, text="Main Menu", font=("Arial", 20))
        label.pack(pady=20)

        btn_data_entry = ttk.Button(self, text="Go to Data Entry", 
                                    command=lambda: controller.show_frame(DataEntry))
        btn_data_entry.pack(pady=10)

        btn_graph_display = ttk.Button(self, text="Go to Graph Display", 
                                       command=lambda: controller.show_frame(GraphDisplay))
        btn_graph_display.pack(pady=10)


class DataEntry(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = ttk.Label(self, text="Data Entry", font=("Arial", 20))
        label.pack(pady=20)

        self.entry = ttk.Entry(self)
        self.entry.pack(pady=10)

        btn_submit = ttk.Button(self, text="Submit", command=self.submit_data)
        btn_submit.pack(pady=5)

        btn_back = ttk.Button(self, text="Back to Main Menu", 
                              command=lambda: controller.show_frame(MainMenu))
        btn_back.pack(pady=10)

    def submit_data(self):
        data = self.entry.get()
        print(f"Data submitted: {data}")  # later you can insert into model or file


class GraphDisplay(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = ttk.Label(self, text="Graph Display", font=("Arial", 20))
        label.pack(pady=20)

        # Placeholder: You can later embed matplotlib here
        btn_show_graph = ttk.Button(self, text="[Graph Placeholder]", command=self.show_graph)
        btn_show_graph.pack(pady=10)

        btn_back = ttk.Button(self, text="Back to Main Menu", 
                              command=lambda: controller.show_frame(MainMenu))
        btn_back.pack(pady=10)

    def show_graph(self):
        print("Graph would be displayed here.")  # Later: matplotlib FigureCanvasTkAgg


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()

