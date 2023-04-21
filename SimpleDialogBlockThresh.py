import tkinter as tk
import tkinter.simpledialog as sd

# Création d'une sous-classe de tkinter.simpledialog.Dialog pour la boîte de dialogue personnalisée
class MyDialog(sd.Dialog):
    def __init__(self, parent, title="My Dialog"):
        self.thresh = 0.0
        self.taille_bloc = 0
        self.filename = ""
        sd.Dialog.__init__(self, parent, title)

    def body(self, master):
        tk.Label(master, text="Threshold:").grid(row=0, sticky=tk.W)
        tk.Label(master, text="Block sizes:").grid(row=1, sticky=tk.W)
        tk.Label(master, text="Filename:").grid(row=2, sticky=tk.W)

        self.threshentry = tk.Spinbox(master, from_=0, to=10, increment=0.1, format="%.1f")
        self.number_var = tk.IntVar()
        self.tailleBlockEntry = tk.Spinbox(master, from_=0, to=64, increment=2, textvariable=self.number_var)
        self.filentry = tk.Entry(master)

        self.threshentry.grid(row=0, column=1)
        self.tailleBlockEntry.grid(row=1, column=1)
        self.filentry.grid(row=2, column=1)

        return self.threshentry # focus on name entry field

    def apply(self):
        self.thresh = float(self.threshentry.get())
        self.taille_bloc = self.number_var.get()
        self.filename = self.filentry.get()

    # Fonction pour ouvrir la boîte de dialogue personnalisée
    def open_dialog(self):
        d = MyDialog(self)
        print("Threshold:", d.thresh)
        print("Block size:", d.taille_bloc)
        print("Filename:", d.filename)