import tkinter as tk
import tkinter.simpledialog as sd

# Création d'une sous-classe de tkinter.simpledialog.Dialog pour la boîte de dialogue personnalisée
class SigmaThreshDialog(sd.Dialog):
    def __init__(self, parent, title="My Dialog"):
        self.thresh = 0.0
        self.sigma = 0
        self.minMatch = 0
        self.filename = ""
        sd.Dialog.__init__(self, parent, title)

    def body(self, master):
        tk.Label(master, text="Threshold:").grid(row=0, sticky=tk.W)
        tk.Label(master, text="Max sigma:").grid(row=1, sticky=tk.W)
        tk.Label(master, text="Min match:").grid(row=2, sticky=tk.W)
        tk.Label(master, text="Filename:").grid(row=3, sticky=tk.W)

        self.threshentry = tk.Spinbox(master, from_=0, to=10, increment=0.01, format="%.2f")
        self.number_var = tk.IntVar()
        self.SigmaEntry = tk.Spinbox(master, from_=0, to=64, increment=2, textvariable=self.number_var)
        self.number_match = tk.IntVar()
        self.MatchEntry = tk.Spinbox(master, from_=0, to=64, increment=2, textvariable=self.number_match)
        self.filentry = tk.Entry(master)

        self.threshentry.grid(row=0, column=1)
        self.SigmaEntry.grid(row=1, column=1)
        self.MatchEntry.grid(row=2, column=1)
        self.filentry.grid(row=3, column=1)

        return self.threshentry # focus on name entry field

    def apply(self):
        self.thresh = float(self.threshentry.get())
        self.sigma = self.number_var.get()
        self.minMatch = self.number_match.get()
        self.filename = self.filentry.get()

    # Fonction pour ouvrir la boîte de dialogue personnalisée
    def open_dialog(self):
        d = SigmeThreshDialog(self)
        print("Threshold:", d.thresh)
        print("Max sigma:", d.sigma)
        print("Min match:", d.minMatch)
        print("Filename:", d.filename)