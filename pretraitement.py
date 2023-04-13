import cv2
import matplotlib.pyplot as plt

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



    