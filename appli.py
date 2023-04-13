import cv2
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import os
from PIL import Image
from SIFT import SiftClustering
from LBP_class import LocBinPatt
from pretraitement import PreProcessing

class App:

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.master.title("Simple menu")

        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Exit", command=self.onExit)
        menubar.add_cascade(label="File", menu=fileMenu)

    def onExit(self):
        self.quit()


def main():

    root = Tk()
    root.geometry("1500x600")
    app = App()
    root.mainloop()


if __name__ == '__main__':
    main()