import tkinter as tk
import tkinter.simpledialog as sd

# Création d'une sous-classe de tkinter.simpledialog.Dialog pour la boîte de dialogue personnalisée
class mydialog(sd.Dialog):
    def __init__(self, parent, title="My Dialog"):
        self.block = 0
        self.filename = ""
        sd.Dialog.__init__(self, parent, title)

    def body(self, master):
        tk.Label(master, text="Block sizes:").grid(row=0, sticky=tk.W)
        tk.Label(master, text="Filename:").grid(row=1, sticky=tk.W)

        self.number_var = tk.IntVar()
        self.blockentry = tk.Spinbox(master, from_=0, to=64, increment=2, textvariable=self.number_var)
        self.filentry = tk.Entry(master)

        self.blockentry.grid(row=0, column=1)
        self.filentry.grid(row=1, column=1)

        return self.blockentry # focus on name entry field

    def apply(self):
        self.block = self.number_var.get()
        self.filename = self.filentry.get()

    # Fonction pour ouvrir la boîte de dialogue personnalisée
    def open_dialog(self):
        d = mydialog(self)
        print("Block size:", d.block)
        print("Filename:", d.filename)