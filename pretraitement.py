import cv2

class PreProcessing:

    @classmethod
    def ndg(self, image):
        image_ndg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image_ndg

    @classmethod
    def contours(self, image):
        image_ndg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(image_ndg, 128, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        return image



    