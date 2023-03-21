from collections import Counter

import matplotlib.pyplot as plt
import cv2
import imutils
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

#Fonction permettant de lire l'image à traiter
def readImage(imagePath):
    return cv2.imread(imagePath)

#Permet d'afficher l'image à l'écran
def showImage(image):
    image = imutils.resize(image, width=512, height=512)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Extraction de caracteristiques grace à l'algorithme SIFT
def caracteristicsExtraction(image):
    imageNdg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(imageNdg, None)#recupere les points clés de l'images, et calcule les descripteurs SIFT qui representent les caracteristique propres de chaque partie de l'image
    return kp, desc

#Mise en correspondance des points clés de l'image
def matchingPoints(kp, desc):
    norm = cv2.NORM_L2 #donne le type de distance calculer entre les points pour les faire correspondre
    bruteForceMatcher = cv2.BFMatcher(norm)#Objet
    k = 20
    matches = bruteForceMatcher.knnMatch(desc, desc, k)
    ratio = 0.5
    good_match_1 = []
    good_match_2 = []

    for match in matches:
        k = 1

        while match[k].distance < ratio * match[k + 1].distance:
            k += 1

            for i in range(1, k):
                if pdist(np.array([kp[match[i].queryIdx].pt, kp[match[i].trainIdx].pt]),
                         "euclidean") > 10:
                    good_match_1.append(kp[match[i].queryIdx])
                    good_match_2.append(kp[match[i].trainIdx])

    points_1 = [match.pt for match in good_match_1]
    points_2 = [match.pt for match in good_match_2]

    if len(points_1) > 0:
        points = np.hstack((points_1, points_2))  # column bind
        unique_points = np.unique(points, axis=0)  # remove any duplicated points
        return np.float32(unique_points[:, 0:2]), np.float32(unique_points[:, 2:4])
    else:
        return None, None

def clustering(points_1, points_2, metric, threshold):
    points = np.vstack((points_1, points_2))  # vertically stack both sets of points (row bind)
    dist_matrix = pdist(points, metric='euclidean')  # obtain condensed distance matrix (needed in linkage function)
    Z = hierarchy.linkage(dist_matrix, metric)

    # perform agglomerative hierarchical clustering
    cluster = hierarchy.fcluster(Z, t=threshold, criterion='inconsistent', depth=4)
    # filter outliers
    #cluster, points = filterOutliers(cluster, points)

    n = int(np.shape(points)[0] / 2)
    return cluster, points[:n], points[n:]

def plotImage(img, p1, p2, C):
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

    plt.savefig("ecureuil_detection.png", bbox_inches='tight', pad_inches=0)
    plt.clf()


def detectCopyMove(image):
    kp, desc = caracteristicsExtraction(image)
    p1, p2 = matchingPoints(kp, desc)
    # showImage(image)x

    if p1 is None:
        # print("No tampering was found")
        return False

    clusters, p1, p2 = clustering(p1, p2, 'ward', 2.2)

    if len(clusters) == 0 or len(p1) == 0 or len(p2) == 0:
        # print("No tampering was found")
        return False

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plotImage(image, p1, p2, clusters)
    return True

def main():
    image = cv2.imread('images/moutons_truquee.jpeg')
    detectCopyMove(image)

if __name__ == "__main__":
    main()
