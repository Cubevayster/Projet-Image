from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import tkinter.simpledialog as sd
from SIFT import SiftClustering
from LBP_class import LocBinPatt
from pretraitement import PreProcessing
from SimpleDialog import mydialog
from simpledialog import MyDialog
from SimpleDialogThresh import threshdialog
import cv2
from PIL import Image
from Analyse import *


class MyWindow(Tk):

    def __init__(self):
        Tk.__init__(self)
        self.create_menu_bar()

        self.geometry("1500x600")
        self.title("DetectApp")
        self.configure(bg = "black")
        self.file = " "
        self.output = " "

        self.sift_vp = 0
        self.sift_vn = 0
        self.sift_fp = 0
        self.sift_fn = 0
        self.sift_clean = True

        self.lbp_vp = 0
        self.lbp_vn = 0
        self.lbp_fp = 0
        self.lbp_fn = 0
        self.lbp_clean = True

    def create_menu_bar(self):
        menu_bar = Menu(self)

        menu_file = Menu(menu_bar, tearoff=0)
        menu_file.add_command(label="New", command=self.do_something)
        menu_file.add_command(label="Open", command=self.open_file)
        menu_file.add_command(label="Save", command=self.do_something)
        menu_file.add_separator()
        menu_file.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=menu_file)

        menu_edit = Menu(menu_bar, tearoff=0)
        menu_edit.add_command(label="Undo", command=self.do_something)
        menu_edit.add_separator()
        menu_edit.add_command(label="Copy", command=self.do_something)
        menu_edit.add_command(label="Cut", command=self.do_something)
        menu_edit.add_command(label="Paste", command=self.do_something)
        menu_bar.add_cascade(label="Edit", menu=menu_edit)

        menu_help = Menu(menu_bar, tearoff=0)
        menu_help.add_command(label="About", command=self.do_about)
        menu_bar.add_cascade(label="Help", menu=menu_help)

        menu_process = Menu(menu_bar, tearoff=0)
        menu_process.add_command(label="Detect contours", command=self.contours)
        menu_process.add_command(label="Threshold", command=self.contours)
        menu_process.add_command(label="Grayscale", command=self.ndg)
        menu_process.add_command(label="Blocks", command=self.block)
        menu_bar.add_cascade(label = "Image processing", menu=menu_process)

        menu_detect = Menu(menu_bar, tearoff=0)
        menu_detect.add_command(label="Sift clustering", command=self.siftclustering)
        menu_detect.add_command(label="LBP operator", command=self.lbp)
        menu_detect.add_command(label="Keypoints", command=self.contours)
        menu_bar.add_cascade(label = "Forgeries detection", menu=menu_detect)

        menu_analyse = Menu(menu_bar, tearoff=0)
        menu_analyse.add_command(label="Sift data", command=self.siftclustering_analysis)
        #menu_analyse.add_command(label="LBP data", command=self.lbp_analysis)
        #menu_analyse.add_command(label="Sift resultats", command=self.siftclustering_analysis_res)
        #menu_analyse.add_command(label="LBP resultats", command=self.lbp_analysis_res)
        #menu_analyse.add_command(label="Sift complxite", command=self.siftclustering_analysis_comp)
        #menu_analyse.add_command(label="LBP complexite", command=self.lbp_analysis_comp)
        menu_bar.add_cascade(label = "Analyse detection results", menu=menu_analyse)

        self.config(menu=menu_bar)

    def open_file(self):
        self.file = askopenfilename(title="Choose the file to open",
                               filetypes=[("PNG image", ".png")])
        if self.file:
            self.charge_image(self.file, 10, 10)


    def charge_image(self, image_path, x, y):
        global image
        image = Image.open(image_path)
        canvas = Canvas(self, width=image.width, height=image.height)
        image = PhotoImage(file = image_path)
        canvas.place(x=x, y=y)
        affichage = canvas.create_image(0, 0, image = image, anchor=NW)
        canvas.image = image

    def siftclustering(self):
        dialog = threshdialog(self)
        answer = messagebox.askyesnocancel("Question", "Do you want an analysis about the results?")
        if dialog.filename is None:
            return None
        elif dialog.filename is not None and not answer:  
            self.output = dialog.filename  
            image = cv2.imread(self.file)
            sift = SiftClustering()
            sift.detectCopyMove(image, self.output, dialog.threshold)
            self.charge_image(self.output, 550, 10)
        else:
            self.output = dialog.filename
            image = cv2.imread(self.file)
            sift = SiftClustering()
            self.last_sift_res, self.exec_time, self.mem_used, self.peak_mem = measure(sift.detectCopyMove, image, self.output, dialog.threshold)
            self.charge_image(self.output, 550, 10)
            self.update_sift(self.last_sift_res, answer)
            write_result("data_sift.json", self.sift_clean, dialog.threshold, self.sift_vp, self.sift_vn, self.sift_fn, self.sift_fp, self.exec_time, self.mem_used, self.peak_mem)
            self.sift_clean = False
    
    def update_sift(self, res, expected):
        if res == expected and res == True : self.sift_vp += 1
        elif res == expected and res == False : self.sift_vn += 1
        elif res != expected and res == True : self.sift_fp += 1
        elif res != expected and res == False : self.sift_fn += 1

    def siftclustering_analysis(self):
        plot_from_file("data_sift.json")

    def do_something(self):
        print("Menu clicked")
    
    def contours(self):
        filename = sd.askstring("Nom du fichier", "Entrer un nom de fichier", parent=self, show='')
        if filename is None:
            return None
        else:  
            self.output = filename  
            image = cv2.imread(self.file)
            Contours = PreProcessing()
            Contours.contours(image, self.output)
            self.charge_image(self.output, 550, 10)

    def ndg(self):
        filename = sd.askstring("Nom du fichier", "Entrer un nom de fichier", parent=self, show='')
        if filename is None:
            return None
        else:  
            self.output = filename  
            image = cv2.imread(self.file)
            NDG = PreProcessing()
            NDG.ndg(image, self.output)
            self.charge_image(self.output, 550, 10)

    def block(self):
        dialog = mydialog(self)
        if dialog.filename is None:
            return None
        else:  
            self.output = dialog.filename  
            LBP = LocBinPatt()
            LBP.compute_and_draw_grid(self.file, self.output, dialog.block)
            self.charge_image(self.output, 550, 10)
    
    def lbp(self):
        dialog = MyDialog(self)
        answer = messagebox.askyesnocancel("Question", "Do you want an analysis about the results?")
        if dialog.filename is None:
            return None
        else:  
            self.output = dialog.filename  
            LBP = LocBinPatt()
            matches = LBP.compare_lbp_desc(self.file, dialog.thresh, dialog.taille_bloc)
            LBP.mark_copy_moved_regions(matches, self.output, dialog.taille_bloc)
            self.charge_image(self.output, 550, 10)

    
    def do_about(self):
        messagebox.showinfo("My title", "My message")


window = MyWindow()
window.mainloop()