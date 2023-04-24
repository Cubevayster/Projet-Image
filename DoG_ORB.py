from skimage.feature import blob_dog, match_descriptors
from math import sqrt
import cv2
import numpy as np
import scipy
from scipy import ndimage
from scipy.spatial import distance
import glob
import os
import math


class ORB_DOG:

    orb = cv2.ORB_create(1000) #Instance de la classe ORB
    matcher = cv2.DescriptorMatcher_create(
        cv2.DescriptorMatcher_BRUTEFORCE_HAMMING) #Matching des descripteurs par brute force grace a la distance de hamming

    @classmethod
    def sobel(self, image):
        inputImage = image.astype(int)
        dx = ndimage.sobel(inputImage, 1) #Application du filtre de sobel selon lhorizontale
        dy = ndimage.sobel(inputImage, 0)#Application du filtre de sobel selon la verticale
        mag = np.hypot(dx, dy) #Calcul de la magnitude de limage (force des gradients)
        mag *= 255.0 / np.max(mag) #Normalisation
        sobelImage = np.uint8(mag)
        return sobelImage

    @classmethod
    #Soustraction de deux images lissees avec des filtres gaussiens
    #Met en evidence les variations dintentiste locales de limage
    def dog(self, greyImage, max_sigma, threshold):
        blobs_dog = blob_dog(greyImage, max_sigma=max_sigma, threshold=threshold) #Calcul des blob (zones sombres ou claires dans image)
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
        return blobs_dog

    @classmethod
    def show(self, blobs_all):
        blob_area = []
        blobs_list = [blobs_all]
        for blobs in blobs_list:
            for blob in blobs:
                y, x, r = blob
                area = [y, x, r]
                if 2*r > 1:
                    blob_area.append(area)
        return blob_area

    @classmethod
    def detectCopyMove(self, image, max_sigma, threshold, min_match_count, output_path):
        sobelImage = self.sobel(image) #Applique le filtre de sobel sur limage
        sobelImageGrey = cv2.cvtColor(sobelImage, cv2.COLOR_BGR2GRAY) #Met limage apres filtre en nuances de gris
        imageGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Met limage originale en nuances de gris

        blobs_all = self.dog(sobelImageGrey, max_sigma, threshold) #Calcul des blobs
        output = self.show(blobs_all)

        imageClone = image.copy()
        key, des = self.orb.detectAndCompute(imageGrey, None) #Calcul des keypoints et descripteurs ORB

        src = np.array([]).reshape(-1, 1, 2)
        dst = np.array([]).reshape(-1, 1, 2)

        liste1 = []
        for blob in range(0, len(output)):
            bloby, blobx, blobr = output[blob] #Recupere les coordonnees de chaque blob ainsi que leur rayon
            cv2.circle(imageClone, (int(blobx), int(bloby)), int(blobr), (255, 0, 0), 1) #Dessine chaque blob sur image resultat
            liste2 = []
            kp1 = []
            ds1 = []
            liste3 = []
            index = 0

            #Parcours de lensemble des keypoints et descripteurs
            for k,d in zip(key, des):
                #Verifie si le point cle est a linterieur du blob
                if (k.pt[0] - blobx)**2 + (k.pt[1] - bloby)**2 <= (blobr**2):
                    liste2.append(index)
                    kp1.append(k) #Stockage des keypoints respectant la condition
                    ds1.append(d) #Stockage des descripteurs respectant la condition
                index += 1

            #Si la liste nest pas vide
            if liste2:
                kp2 = np.delete(key, liste2, axis=0) #Suppression dans la liste des points cles, des points cle appartenant au blob courant
                ds2 = np.delete(des, liste2, axis=0) #Suppression dans la liste des descripteurs, des descripteurs appartenant au blob courant
                nnMatches = self.matcher.knnMatch(np.array(ds1), ds2, 2) #Compare les descripteurs appartenant au blob a ceux exterieurs au blob et donne une liste de paire de match possibles

                goodMatch = []

                nnMatchRatio = 0.6

                #Parcours de lensemble des match obtenus precedemment
                for m,n in nnMatches:
                    #Filtre les match pour eliminer les mauvais ou ambigus
                    if m.distance < nnMatchRatio * n.distance:
                        goodMatch.append(m)

                #Pour un certain nombre de match obtenus, on considere une zone copiee-deplacee
                if len(goodMatch) > min_match_count:
                    #Recuperation des coordonnees des keypoints correspondant
                    srcPoints = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
                    dstPoints = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
                    src = np.array(srcPoints).ravel()
                    dst = np.array(dstPoints).ravel()

                    ps = np.array(src).reshape((-1, 2))
                    pd = np.array(dst).reshape((-1, 2))

                    #Dessin des keypoints et ligne les reliant pour identifier la zone copiee deplacee
                    for k1, k2 in zip(ps, pd):
                        cv2.circle(imageClone, (int(k1[0]), int(k1[1])), 4, (255, 0, 0), -1)
                        cv2.circle(imageClone, (int(k2[0]), int(k2[1])), 4, (0, 0, 255), -1)
                        cv2.line(imageClone,(int(k1[0]),int(k1[1])),(int(k2[0]),int(k2[1])),(0,255,0),2)   
                        

        cv2.imwrite(output_path, imageClone)
