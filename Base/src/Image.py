import sys
import math
import time
import psutil
import sys
import os
import time
import psutil

#importer matplotlib, scikit-learn et scikit-image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage.measure import label

format_couleur = ["PNG","JPEG","JPG","PPM"]
format_gris = ["PPM","BPM"]

class Img:
    def __init__(self, path):
        self.path = path
        self.image = Image.open(path)
        self.format = self.image.format
        self.tableau = np.array(self.image)
        self.tableau2D = np.reshape(self.image,(-1, 3))
        self.pgm = self.image.convert('L')
        self.tableauPGM = np.array(self.pgm)

    def disp(self):
        print(f"Imagetableau object of format {self.format} and shape {self.tableau.shape}")

    def show(self):
        self.image.show()

    def graph(self, name=None):
        plt.imshow(self.tableau)
        plt.title(name)
        plt.show()

    def save(self, path):
        self.image.save(path)

    def R(self):
        if self.format in format_couleur:
            return self.tableau[:, :, 0]
        else:
            print("pas bon format")

    def G(self):
        if self.format in format_couleur:
            return self.tableau[:, :, 1]
        else:
            print("pas bon format")

    def B(self):
        if self.format in format_couleur:
            return self.tableau[:, :, 2]
        else:
            print("pas bon format")

    def RGB_to_YCrCb(self):
        if self.format in format_couleur:
            R = self.R()
            G = self.G()
            B = self.B()
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            Cr = 0.713 * (R - Y) + 128
            Cb = 0.564 * (B - Y) + 128
            self.tableau = np.dstack((Y, Cr, Cb))
        else:
            print("pas bon format")

    #Utile pour les régions apparement
    def Y(self):
        if self.tableau.shape[2] == 3:
            return self.tableau[:, :, 0]
        else:
            print("pas bon format")

    def Cr(self):
        if self.tableau.shape[2] == 3:
            return self.tableau[:, :, 1]
        else:
            print("pas bon format")

    def Cb(self):
        if self.tableau.shape[2] == 3:
            return self.tableau[:, :, 2]
        else:
            print("pas bon format")

    def YCrCb_to_RGB(self):
        if self.tableau.shape[2] == 3:
            Y = self.Y()
            Cb = self.Cb()
            Cr = self.Cr()
            R = Y + 1.403 * Cr
            G = Y - 0.344 * Cr - 0.714 * Cb
            B = Y + 1.773 * Cb
            self.tableau = np.dstack((R, G, B))
        else:
            print("pas bon format")

    def Kmeans(k):
        Kmeans = KMeans(n_clusters=k, random_state=0).fit(self.tableau2D)
        Reconstruc_seg_image = Kmeans.cluster_centers_[Kmeans.labels_]
        Reconstruc_seg_image = np.reshape(Reconstruc_seg_image, self.image.shape)
        return Reconstruc_seg_image

    def Kmeans_compare(Reference, k):
        IMG_original = self.cluster(k)
        IMG_reference = Reference.cluster(k)

        MoyO = np.reshape(KMeans(n_clusters=k, random_state=0).fit(np.reshape(IMG_original, (-1, 3))).cluster_centers_, (1, k, 3))
        MoyR = np.reshape(KMeans(n_clusters=k, random_state=0).fit(np.reshape(IMG_reference, (-1, 3))).cluster_centers_, (k, 1, 3))

        d_euclide = np.linalg.norm(MoyO - MoyR, axis=2)
        return np.mean(d_euclide)

    def cluster(self, *args, **kwargs):
        if k == 1 :
            return self.KMeans(kwargs["kmean"])

    def compare(self, option, *args):
        if option == 1 :
            return Kmeans_compare(kwargs["reference"], kwargs["kmean"])

    def falsification(self) :
        falsifie = True
        return falsifie

    #Cette fonction suit l'algortihme de points d'intérêts par segmentation spatiale de régions
    #Un peu chaud à faire et incomplète sauf la base de la base
    #Rien ne garanti que 'la base' est une bonne base, franchement j'ai pas confiance
    #Les accès à la classe sont bons mais encore faut bien s'en servir au bon moment
    #Pas sur avec un code incomplet
    def region(self, threshold, min_size):
        # Niveaux de gris
        self.tableauPGM
        # Carte de segmentation
        Seg = np.zeros_like(self.tableauPGM)
        Current = 1
        # Parcourt + Segmentation regions
        for i in range(self.tableauPGM[0]):
            for j in range(self.tableauPGM[1]):
                # Pixel pas trouvé
                    # Définir région intiale courrante
                    region = []
                    region.add((i, j))
                    # Bornes
                    limit = 0
                    # Parcourir les voisins pour trouver le reste de la région
                    # Faire pile pixel --> region.pop()
                    while len(region) > 0:
                        pixel = region.pop()
                        voisins = []
                        for v in voisins:
                            # Si pixel in region et pas deja compte
                            compte = True
                            In = False
                            if (In and not compte):
                                # Si pixel ok et pas hors limites
                                Ok = False
                                hors = True
                                if Ok and not hors:
                                    region.add(v)
                                    Seg[v]=Current
                    # Si region est ok passer à la suivante
                    region_ok = True
                    if region_ok : Current += 1
        # Étiqueter régions
        Map = label(Seg)
        
        return Map
