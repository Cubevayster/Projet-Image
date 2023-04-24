import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np

class PreProcessing:

    @classmethod
    def ndg(self, image, output_path):
        image_ndg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_path, image_ndg)

    @classmethod
    def contours(self, image, output_path):
        image_ndg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Appliquer un flou gaussien pour réduire le bruit
        blurred = cv2.GaussianBlur(image_ndg, (5, 5), 0)
        # Appliquer l'algorithme de détection de contours Canny
        edged = cv2.Canny(blurred, 30, 150)
        cv2.imwrite(output_path, edged)

    @classmethod
    def keypoints(self, image, output_path):
        imageNdg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Instance de la classe sift pour pour appliquer lalgorithme
        sift = cv2.SIFT_create()
        #Calcul des keypoints et descripteurs sift
        kp, desc = sift.detectAndCompute(imageNdg, None)
        output_image = cv2.drawKeypoints(imageNdg, kp, 0, (0, 0, 255),
                                 flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(output_path, output_image)
        return kp, desc

    @classmethod
    def sobel(self, image, output_path):
        inputImage = image.astype(int)
        dx = ndimage.sobel(inputImage, 1) #Application du filtre de sobel selon lhorizontale
        dy = ndimage.sobel(inputImage, 0)#Application du filtre de sobel selon la verticale
        mag = np.hypot(dx, dy) #Calcul de la magnitude de limage (force des gradients)
        mag *= 255.0 / np.max(mag) #Normalisation
        sobelImage = np.uint8(mag)
        cv2.imwrite(output_path, sobelImage)