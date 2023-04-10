import cv2
from tkinter import *
from tkinter import filedialog
import os
from PIL import Image
from SIFT import SiftClustering

class Application():

    import_file = " "
    sift_file =  "data/sift_detecion.png"
    

    def __init__(self):
        super().__init__()

    @classmethod
    def Import(self):
        global file_to_display
        self.import_file = filedialog.askopenfilename(title = "Import a file",
                                                defaultextension = ".jpg",
                                                filetypes = [("JPG File", ".jpg"), ("JPEG File", ".jpeg"), ("PNG File", ".png")]
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
        
        




app = Application()

window = Tk()
window.title("Copy-move forgery detection")
window.configure(bg ="black")
window.geometry("1000x600")

OpenDirectoryButton = Button(window, text = "Import file", bg = "grey", fg = "white", command = app.Import)
OpenDirectoryButton.grid(row=2, column=1)

SiftClusteringButton = Button(window, text = "Copy-move forgery detection by Sift Clustering", bg = "grey", fg = "white", command=app.applySift)
SiftClusteringButton.grid(row=2, column=2)

LBPButton = Button(window, text = "Copy-move forgery detection by LBP", bg = "grey", fg = "white")
LBPButton.grid(row=2, column=3)

gifsdict={} 





window.mainloop()