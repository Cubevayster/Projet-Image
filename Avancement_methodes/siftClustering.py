import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

class SiftClustering:

    @classmethod
    def readImage(self, path):
        img = cv2.imread(path)
        return img

    @classmethod
    def showImage(self, img, name):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @classmethod
    def caracteristicsExtraction(self, img):
        imageNdg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, desc = sift.detectAndCompute(imageNdg, None)
        return kp, desc


    @classmethod
    def matchingPoints(self, kp, desc):
        norm = cv2.NORM_L2  # donne le type de distance calculer entre les points pour les faire correspondre
        bruteForceMatcher = cv2.BFMatcher(norm)  # Objet
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

    @classmethod
    def clustering(self, points_1, points_2, metric, threshold):
        points = np.vstack((points_1, points_2))  # vertically stack both sets of points (row bind)
        dist_matrix = pdist(points, metric='euclidean')  # obtain condensed distance matrix (needed in linkage function)
        Z = hierarchy.linkage(dist_matrix, metric)

        # perform agglomerative hierarchical clustering
        cluster = hierarchy.fcluster(Z, t=threshold, criterion='inconsistent', depth=4)

        n = int(np.shape(points)[0] / 2)
        return cluster, points[:n], points[n:]

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
    def detectCopyMove(self, image, title):
        kp, desc = self.caracteristicsExtraction(image)
        p1, p2 = self.matchingPoints(kp, desc)
        # showImage(image)x

        if p1 is None:
            # print("No tampering was found")
            return False

        clusters, p1, p2 = self.clustering(p1, p2, 'ward', 2.2)

        if len(clusters) == 0 or len(p1) == 0 or len(p2) == 0:
            # print("No tampering was found")
            return False

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.plotImage(self, image, p1, p2, clusters, title)
        return True

