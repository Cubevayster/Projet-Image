import cv2
from tkinter import *
from tkinter import filedialog
import os
from PIL import Image
from SIFT import SiftClustering

class Application():

    import_file = " "
    sift_file =  "data/sift_detecion.png"
    lbp_file = "data/lbp_detection.png"
    

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
    def applyLbp(self):
        loc_bin_patt = LBP()
        img = loc_bin_patt.readImage(self.import_file)
        loc_bin_patt.display_forgeries(img, self.lbp_file)

        result_lbp_file = Image.open(self.lbp_file)
        result_lbp_image_holder = Canvas(window, width=result_lbp_file.width, height=result_lbp_file.height)
        result_lbp_image_holder.place(x=550, y=50)
        result_lbp_file = PhotoImage(file = self.lbp_file)
        image_lbp_result = result_lbp_image_holder.create_image(0, 0, image = result_lbp_file, anchor=NW)

        gifsdict[img] = result_lbp_file
        LBP.configure(image=result_lbp_file) 




app = Application()

window = Tk()
window.title("Copy-move forgery detection")
window.configure(bg ="black")
window.geometry("1000x600")

OpenDirectoryButton = Button(window, text = "Import file", bg = "grey", fg = "white", command = app.Import)
OpenDirectoryButton.grid(row=2, column=1)

SiftClusteringButton = Button(window, text = "Copy-move forgery detection by Sift Clustering", bg = "grey", fg = "white", command=app.applySift)
SiftClusteringButton.grid(row=2, column=2)

LBPButton = Button(window, text = "Copy-move forgery detection by LBP", bg = "grey", fg = "white", command=app.applyLbp)
LBPButton.grid(row=2, column=3)

SiftButton = Button(window, text = "View keypoints", bg = "grey", fg = "white")
SiftButton.grid(row=2, column=4)

SiftClusterButton = Button(window, text = "View keypoints clusters", bg = "grey", fg = "white")
SiftClusterButton.grid(row=2, column=5)

gifsdict={} 





window.mainloop()