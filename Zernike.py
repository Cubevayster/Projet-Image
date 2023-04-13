import cv2
import numpy as np
from scipy.special import comb

# Définition de la fonction pour calculer les moments de Zernike
def zernike_moments(img, degree):
    moments = []
    height, width = img.shape[:2]
    radius = min(height, width) / 2.0
    
    # Centrage de l'image
    y, x = np.indices(img.shape)
    x = x - (width-1)/2.0
    y = (height-1)/2.0 - y
    
    # Calcul des moments de Zernike
    for n in range(degree+1):
        for m in range(n+1):
            if (n-m) % 2 == 0:
                rnm = np.sqrt(comb(n, m)) * (radius**m) * (np.sum(x**2 + y**2 <= radius**2) / np.pi)
                thetanm = m * np.arctan2(y, x)
                cnm = np.sum(img * rnm * np.cos(m*thetanm)) / np.sum(rnm**2)
                snm = np.sum(img * rnm * np.sin(m*thetanm)) / np.sum(rnm**2)
                moments.append(cnm + 1j*snm)
                
    return moments

# Chargement de l'image en niveaux de gris
img = cv2.imread('moutons_truquee.jpeg', cv2.IMREAD_GRAYSCALE)

# Seuillage de l'image
threshold = 127
img_threshold = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]

# Calcul des moments de Zernike de l'image seuillée
degree = 10
moments = zernike_moments(img_threshold, degree)

# Détection des zones de copiés déplacés à partir des moments de Zernike
threshold_ratio = 0.9
threshold_moments = threshold_ratio * max([abs(moment) for moment in moments])
copied_regions = []
for i, moment in enumerate(moments):
    if abs(moment) >= threshold_moments:
        copied_regions.append(i)

# Mise en évidence des zones de copiés déplacés dans l'image
img_highlighted = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for i, moment in enumerate(moments):
    if i in copied_regions:
        cv2.drawContours(img_highlighted, [np.int0(cv2.boxPoints(cv2.minAreaRect(np.argwhere(img_threshold))))], 0, (0, 0, 255), 2)

# Enregistrement de l'image résultante avec les zones de copiés déplacés mises en évidence
cv2.imwrite('data/zernike.png', img_highlighted)
