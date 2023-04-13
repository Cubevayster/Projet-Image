import cv2
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import os
from PIL import Image
from SIFT import SiftClustering
from LBP_class import LocBinPatt
from pretraitement import PreProcessing

class Application():

    import_file = " "
    sift_file =  "data/sift_detecion.png"
    lbp_file = "data/image_lbp.png"
    bloc_file = "data/image_blocs.png"
    kp_file = "data/image_blocs_keypoints.png"
    clust_file = "data/image_clusters.png"
    

    def __init__(self):
        super().__init__()

    @classmethod
    def Import(self):
        global file_to_display
        self.import_file = filedialog.askopenfilename(title = "Import a file",
                                                defaultextension = ".png",
                                                filetypes = [("PNG File", ".png")]
                                                )   

    
        file_to_display = Image.open(self.import_file)
        image_holder = Canvas(window, width=file_to_display.width, height=file_to_display.height)
        image_holder.place(x=10, y=50)
        file_to_display = PhotoImage(file = self.import_file)
        image = image_holder.create_image(0, 0, image = file_to_display, anchor=NW)


    @classmethod
    def applySift(self):
        inv_scale_features = SiftClustering()
        img = inv_scale_features.readImage(self.import_file)
        inv_scale_features.detectCopyMove(img, self.sift_file)

        result_file = Image.open(self.sift_file)
        result_image_holder = Canvas(window, width=result_file.width, height=result_file.height)
        result_image_holder.place(x=550, y=50)
        result_file = PhotoImage(file = self.sift_file)
        image_result = result_image_holder.create_image(0, 0, image = result_file, anchor=NW)

        gifsdict[img] = result_file
        SiftClustering.configure(image=result_file) 
        

    @classmethod
    def divide_image(self):
        image_divided = LocBinPatt()
        blocs = image_divided.compute_and_draw_grid(self.import_file)

        image_to_display = Image.open(self.bloc_file)
        image_holder = Canvas(window, width=image_to_display.width, height=image_to_display.height)
        image_holder.place(x=550, y=50)
        image_to_display = PhotoImage(file = self.bloc_file)
        displayed_image = image_holder.create_image(0, 0, image = image_to_display, anchor=NW)

        gifsdict[blocs] = image_to_display
        LocBinPatt.configure(image=image_to_display)

    @classmethod
    def display_keypoints(self):
        image_keypoints = LocBinPatt()
        image_keypoints.compute_and_draw_keypoints(self.import_file)

        image_to_disp = Image.open(self.kp_file)
        image_holder = Canvas(window, width=image_to_disp.width, height=image_to_disp.height)
        image_holder.place(x=550, y=50)
        image_to_disp = PhotoImage(file = self.kp_file)
        displayed_image = image_holder.create_image(0, 0, image = image_to_disp, anchor=NW)

        gifsdict[image_keypoints] = image_to_disp
        LocBinPatt.configure(image=image_to_disp)

    @classmethod
    def display_LBP(self):
        image_lbp = LocBinPatt()
        matches = image_lbp.compare_lbp_desc(self.import_file)
        image_lbp.mark_copy_moved_regions(matches)
        image_to_disp = Image.open(self.lbp_file)
        image_holder = Canvas(window, width=image_to_disp.width, height=image_to_disp.height)
        image_holder.place(x=550, y=50)
        image_to_disp = PhotoImage(file = self.lbp_file)
        displayed_image = image_holder.create_image(0, 0, image = image_to_disp, anchor=NW)

        gifsdict[matches] = image_to_disp
        LocBinPatt.configure(image=image_to_disp)

    @classmethod
    def detection_contours(self):
        contours = PreProcessing()
        image = cv2.imread(self.import_file)
        image = contours.contours(image)
        cv2.imshow('Détection de contours', image)


app = Application()

window = Tk()
window.title("Copy-move forgery detection")
window.configure(bg ="black")
window.geometry("1500x600")

OpenDirectoryButton = Button(window, text = "Import file", bg = "grey", fg = "white", command = app.Import)
OpenDirectoryButton.grid(row=2, column=1)

SiftClusteringButton = Button(window, text = "Copy-move forgery detection by Sift Clustering", bg = "grey", fg = "white", command=app.applySift)
SiftClusteringButton.grid(row=2, column=2)

BlockButton = Button(window, text = "Divide image into blocks", bg = "grey", fg = "white", command = app.divide_image)
BlockButton.grid(row=2, column=3)

KPButton = Button(window, text = "Display image's keypoints", bg = "grey", fg = "white", command = app.display_keypoints)
KPButton.grid(row=2, column=4)

LPBButton = Button(window, text = "Apply LBP operator", bg = "grey", fg = "white", command = app.display_LBP)
LPBButton.grid(row=2, column=5)




gifsdict={} 





window.mainloop()