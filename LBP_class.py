import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from skimage.feature import local_binary_pattern
from scipy.spatial import distance


class LocBinPatt:
    hauteur = 0 #hauteur de limage
    largeur = 0 #largeur de limage
    nb_blocs_hauteur = 0 #Nombre de bloc en hauteur
    nb_blocs_largeur = 0 #Nombre de bloc en largeur
    image = None 
    blocs = []
    keypoint_block = []

    @classmethod
    def decoupe_image_en_blocs(self, image_path, taille_bloc):
        cpt = 0 #Compteur pour les blocs
        self.image = cv2.imread(image_path) #Lecture de limage
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) #Conversion de limage en nuance de gris

        self.hauteur, self.largeur = self.image.shape[:2] #Recuperation de hauteur et largeur de limage

        self.nb_blocs_largeur = int(self.largeur/taille_bloc) #Calcul du nombre de blocs en largeur
        self.nb_blocs_hauteur = int(self.hauteur/taille_bloc) #Calcul du nombre de blocs en hauteur

        for j in range(self.nb_blocs_hauteur):
            for i in range(self.nb_blocs_largeur):
                x = i*taille_bloc
                y = j*taille_bloc

                #Divison de limage en blocs
                bloc = self.image[y:y+taille_bloc, x:x+taille_bloc]

                self.blocs.append(bloc)
                cpt += 1
        return self.blocs

    @classmethod
    def dessiner_grille(self, output_path, taille_bloc):
        grille = np.zeros((self.hauteur, self.largeur), dtype=np.uint8)
        for j in range(self.nb_blocs_hauteur):
            for i in range(self.nb_blocs_largeur):
                x = i*taille_bloc
                y = j*taille_bloc

                grille[y:y+taille_bloc, x:x +
                       taille_bloc] = self.blocs[j*self.nb_blocs_largeur+i]
                cv2.rectangle(grille, (x, y), (x+taille_bloc,
                              y+taille_bloc), (255, 255, 255), 1)

        img_grille = cv2.addWeighted(self.image, 0.5, grille, 0.5, 0)
        cv2.imwrite(output_path, img_grille)

    @classmethod
    def compute_and_draw_grid(self, image_path, output_path, taille_bloc):
        self.blocs = self.decoupe_image_en_blocs(image_path, taille_bloc)
        self.dessiner_grille(output_path, taille_bloc)

    @classmethod
    def compute_and_draw_keypoints(self, image_path):
        img = cv2.imread(image_path) #Lecture de limage

        # Détection des keypoints et des descripteurs avec le détecteur SIFT
        sift = cv2.SIFT_create() #Instance de la classe sift
        kp, des = sift.detectAndCompute(img, None) #Calcul des keypoints et des descripteurs

        # Dessin des keypoints sur l'image
        img_with_keypoints = cv2.drawKeypoints(
            img, kp, None, color=(0, 255, 0), flags=0)
        cv2.imwrite('data/image_blocs_keypoints.png', img_with_keypoints) #Ecrire et enregistrer limage

    @classmethod
    def compute_LBP(self, image_path):
        image = cv2.imread(image_path)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(grey, 8, 1, method='var') #Calcul des descripteurs LBP de limage entiere
        print(lbp)
        cv2.imwrite('data/image_lbp.png', lbp)
        
    @classmethod
    def compute_lbp_per_block(self, image_path):
        lbp_descriptors = []
        for block in self.blocs:
            lbp = local_binary_pattern(block, 8, 1, method='uniform') #Calcul des descripteur LBP par blocs de limage
            hist, _ = np.histogram(lbp, bins=range(0, 8), density=True) #Determination des histogrammes LBP de chaque bloc de limage
            lbp_descriptors.append(hist)
        lbp_descriptors = np.array(lbp_descriptors)
        return lbp_descriptors

    @classmethod
    def compare_lbp_desc(self, image_path, threshold, taille_bloc):
        matches = []
        self.blocs = self.decoupe_image_en_blocs(image_path, taille_bloc) #Decoupe image en blocs
        lbp_desc = self.compute_lbp_per_block(image_path) #Calcul les descripteurs LBP
        for i in range(0, len(self.blocs)):
            for j in range(i+1, len(self.blocs)):
                dist = distance.euclidean(lbp_desc[i], lbp_desc[j]) #Calcul les distances entre les descripteurs
                #Si la distance est inferieure a un certain seul
                if dist < threshold: 
                    #Alors on considere le match
                    matches.append((i,j))
        return matches   

    @classmethod
    def mark_copy_moved_regions(self, matches, output_path, taille_bloc):
        #Pour chaque match trouve
        for match in matches:
            x1, y1 = match
            #encadrer la zone correspondante par un rectangle sur limage resultat
            cv2.rectangle(self.image, (x1*taille_bloc, y1*taille_bloc), (x1*taille_bloc + taille_bloc, y1*taille_bloc + taille_bloc), (0, 0, 255), 2)

        cv2.imwrite(output_path, self.image)
