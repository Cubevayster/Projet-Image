import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

class SiftClustering:

    @classmethod
    #Methode permettant de lire une image
    def readImage(self, path):
        img = cv2.imread(path)
        return img

    @classmethod
    #Methode permettant de montrer une image
    def showImage(self, img, name):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @classmethod
    #Extraction des points clef d une image et de leur descripteurs associes
    def caracteristicsExtraction(self, img):
        #Passage de limage en nuance de gris pour ne travailler que sur un seul canal de couleur
        imageNdg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Instance de la classe sift pour pour appliquer lalgorithme
        sift = cv2.SIFT_create()
        #Calcul des keypoints et descripteurs sift
        kp, desc = sift.detectAndCompute(imageNdg, None)
        return kp, desc


    @classmethod
    #Methode pour faire correspondre les points clef
    def matchingPoints(self, kp, desc):
        norme_L2 = cv2.NORM_L2  #Donne la mesure de distance utilisee pour determiner les matches
        bruteForceMatcher = cv2.BFMatcher(norme_L2)  #Instance de la classe BFMatcher permettant de faire matcher les points selon une certaine distance
        k = 20 #Represente le nombre de meilleurs matches voulus
        matches = bruteForceMatcher.knnMatch(desc, desc, k) #Calcul les matches
        ratio = 0.5 #utile pour filtrer les matches
        goodMatch1 = []
        goodMatch2 = []

        #Parcours de lensemble des matches trouves pour les filtrer
        for match in matches:
            k = 1

            #Tant que la condition est remplie cela signifie que cest un bon match
            while match[k].distance < ratio * match[k + 1].distance:
                k += 1

                for i in range(1, k):
                    #Evite les correspondances erronees en verifiant leloignement spatial en pixel entre les bons match  
                    if pdist(np.array([kp[match[i].queryIdx].pt, kp[match[i].trainIdx].pt]),
                             "euclidean") > 10:
                        #Stockage des bons match dans deux listes
                        goodMatch1.append(kp[match[i].queryIdx])
                        goodMatch2.append(kp[match[i].trainIdx])
        #Recuperation des coordonnees des points clef correspondants aux bons match
        points1 = [match.pt for match in goodMatch1]
        points2 = [match.pt for match in goodMatch2]

        #Si des bons match on ete trouves
        if len(points1) > 0:
            points = np.hstack((points1, points2))
            unique_points = np.unique(points, axis=0) #Suppression des points dupliques
            return np.float32(unique_points[:, 0:2]), np.float32(unique_points[:, 2:4])
        else:
            return None, None

    @classmethod
    def clustering(self, points1, points2, metric, threshold):
        points = np.vstack((points1, points2)) #Concatenation verticale des points dans un seul tableau
        matriceDesDistances = pdist(points, metric='euclidean') #Obtention d une matrice des distances condensee
        Z = hierarchy.linkage(matriceDesDistances, metric) #Clustering hierarchique sur la matrice des distances

        cluster = hierarchy.fcluster(Z, t=threshold, criterion='inconsistent', depth=4) #Assigne chaque point clef a un cluster grace a larbre Z

        n = int(np.shape(points)[0] / 2) #Donne le nombre de points clef
        return cluster, points[:n], points[n:]

    #Ecrit limage avec detection
    def plotImage(self, img, p1, p2, C, title):
        plt.imshow(img)
        plt.axis('off')

        colors = C[:np.shape(p1)[0]]
        plt.scatter(p1[:, 0], p1[:, 1], c=colors, s=30)

        for coord1, coord2 in zip(p1, p2):
            x1 = coord1[0]
            y1 = coord1[1]

            x2 = coord2[0]
            y2 = coord2[1]

            plt.plot([x1, x2], [y1, y2], 'c', linestyle=":")

        plt.savefig(title, bbox_inches='tight', pad_inches=0)
        plt.clf()

    @classmethod
    def detectCopyMove(self, image, title, th):
        kp, desc = self.caracteristicsExtraction(image)
        p1, p2 = self.matchingPoints(kp, desc)

        if p1 is None:
            print("No tampering was found")
            return False

        if th is None : th = float(2.2) #Definit un seuil par defaut
        
        clusters, p1, p2 = self.clustering(p1, p2, 'ward', th)

        if len(clusters) == 0 or len(p1) == 0 or len(p2) == 0:
            print("No tampering was found")
            return False

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.plotImage(self, image, p1, p2, clusters, title)
        return True

