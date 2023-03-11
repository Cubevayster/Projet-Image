import sys
import math
import time
import psutil
import sys
import os
import time
import psutil
import cv2
from scipy.stats import norm

# importer matplotlib, scikit-learn et scikit-image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage.measure import label
from sklearn import svm

format_couleur = ["PNG", "JPEG", "JPG", "PPM"]
format_gris = ["PPM", "BPM"]

default_threshold = 0.5


class Img:
    def __init__(self, path, fal=False):
        self.path = path
        self.image = Image.open(path)
        self.img = cv2.imread(path)
        self.format = self.image.format
        self.tableau = np.array(self.image)
        self.tableau2D = np.reshape(self.image, (-1, 3))
        self.pgm = self.image.convert('L')
        self.tableauPGM = np.array(self.pgm)
        self.data = list(self.image.getdata())
        self.width, self.height = self.image.size
        # test corrélation
        self.NPtableauPGM = cv2.cvtColor(self.tableau, cv2.COLOR_RGB2GRAY)
        self.sift = cv2.SIFT_create()
        self.key_point, self.descriptors = self.sift.detectAndCompute(self.NPtableauPGM, None)

        # test corrélation
        self.falsif = fal

        # learning ou default
        self.d_threshold = 0.5
        self.threshold = svm.SVC()
        self.jeu = None
        self.jeu_etiq = None

    def disp(self):
        print(f"Image tableau object of format {self.format} and shape {self.tableau.shape} from {self.path}")

    def show(self):
        self.image.show()

    def graph(self, name=None):
        plt.imshow(self.tableau)
        plt.title(name)
        plt.show()

    def save(self, path):
        self.image.save(path)

    def histogram(self, bins=256):
        hist, binss = np.histogram(self.tableauPGM, bins=bins, range=(0, 255))
        return hist, binss

    def gauss(self, x, mu=1, sigma=0):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    def drawSift(self, path, isGrey):
        img2 = self.img
        if(isGrey):
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            keyPoints = self.sift.detect(gray, None)
            img2 = cv2.drawKeypoints(gray, keyPoints, img2)
            cv2.imwrite(path, img2)
        else:
            keyPoints = self.sift.detect(img2, None)
            img2 = cv2.drawKeypoints(img2, keyPoints, img2)
            cv2.imwrite(path, img2)



    def plot_histogram(self, bins=256):
        hist, binss = self.histogram(bins=bins)
        plt.bar(range(len(hist)), hist)
        plt.title(f"Histogramme de l'image {self.path}")
        plt.xlabel("Niveau de gris")
        plt.ylabel("Nombre de pixels")
        plt.show()

    def plot_gauss(self, mu=1, sigma=0):
        x = np.linspace(0, 255, 256)
        y = self.gauss(x, mu, sigma)
        plt.plot(x, y)
        plt.title(f"Guass de l'image {self.path}")
        plt.xlabel("Niveau de gris")
        plt.ylabel("Probabilités")
        plt.show()

    def plot_histogramme_couleur(self):
        figure, axis = plt.subplots(1, 3, figsize=(10, 5))
        colors = ['r', 'g', 'b']
        for i, color in enumerate(colors):
            axis[i].hist(self.tableau[..., i].ravel(), bins=256, color=color, alpha=0.5)
            axis[i].set_xlim([0, 256])
            axis[i].set_ylim([0, self.width * self.height * 0.5])
            axis[i].set_title(f'Histogramme {color.upper()}')
        plt.show()

    def plot_gauss_couleur(self):
        figure, axis = plt.subplots(1, 3, figsize=(10, 5))
        colors = ['r', 'g', 'b']
        for i, color in enumerate(colors):
            data = self.tableau[..., i].ravel()
            mean = np.mean(data)
            std = np.std(data)
            x = np.linspace(0, 256, 256)
            axis[i].hist(data, bins=256, color=color, alpha=0.5, density=True)
            axis[i].plot(x, norm.pdf(x, mean, std), color='black', alpha=0.8)
            axis[i].set_xlim([0, 256])
            axis[i].set_ylim([0, 0.03])
            axis[i].set_title(f'Histogramme et gaussienne {color.upper()}')
        plt.show()

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

    # Utile pour les régions apparement
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

    def Kmeans(self, k):
        Kmeans = KMeans(n_clusters=k, random_state=0).fit(self.tableauPGM)

        Reconstruc_seg_image = Kmeans.cluster_centers_[Kmeans.labels_]
        Reconstruc_seg_image = np.reshape(Reconstruc_seg_image, self.tableauPGM.shape)

        segmented_image = Image.fromarray(np.uint8(Reconstruc_seg_image))
        segmented_image = segmented_image.convert('RGB')

        return segmented_image  # , Kmeans.inertia_

    def Kmeans_compare(self, Reference, k):
        IMG_original = self.cluster(k)
        IMG_reference = Reference.cluster(k)

        MoyO = np.reshape(KMeans(n_clusters=k, random_state=0).fit(np.reshape(IMG_original, (-1, 3))).cluster_centers_,
                          (1, k, 3))
        MoyR = np.reshape(KMeans(n_clusters=k, random_state=0).fit(np.reshape(IMG_reference, (-1, 3))).cluster_centers_,
                          (k, 1, 3))

        d_euclide = np.linalg.norm(MoyO - MoyR, axis=2)
        return np.mean(d_euclide)

    def cluster(self, *args, **kwargs):
        if k == 1:
            return self.KMeans(kwargs["kmean"])

    def compare(self, option, *args):
        if option == 1:
            return Kmeans_compare(kwargs["reference"], kwargs["kmean"])

    def train(self, data_jeu, data_etiq):
        self.jeu = np.array([Image.open(data_path).convert('L').flatten() for data_path in data_jeu])
        self.jeu_etiq = np.array(data_etiq)
        self.threshold.fit(self.jeu, self.jeu_etiq)

    def predict(self):
        pgm = self.pgm.flatten()
        return self.threshold.predict([pgm])

    def falsification(self, threshold=default_threshold, ia=False):
        if ia is False:
            if self.descriptors is None:
                self.falsif = False
            else:
                score = np.mean(self.descriptors < threshold)
                if score > 0.5:
                    self.falsif = True
                else:
                    self.falsif = False
        else:
            self.falsif = bool(self.predict())

    # Cette fonction suit l'algortihme de points d'intérêts par segmentation spatiale de régions
    # Un peu chaud à faire et incomplète sauf la base de la base
    # Rien ne garanti que 'la base' est une bonne base, franchement j'ai pas confiance
    # Les accès à la classe sont bons mais encore faut bien s'en servir au bon moment
    # Pas sur avec un code incomplet
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
                                Seg[v] = Current
                # Si region est ok passer à la suivante
                region_ok = True
                if region_ok: Current += 1
        # Étiqueter régions
        Map = label(Seg)

        return Map
