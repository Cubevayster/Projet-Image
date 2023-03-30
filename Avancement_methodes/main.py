import cv2
import numpy as np

image = cv2.imread("images/moutons_truquee.jpeg")
height = image.shape[0]
width = image.shape[1]
channels = image.shape[2]
image_ndg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blockSize = 16 * 16
numberBlocks = (height - 16 + 1) * (width - 16 + 1)

#Détermination des blocks
starting_points_X = []
starting_points_Y = []

def starting_points(size, split_size, overlap):
    points = [0]
    step = int(split_size * (1 - overlap))
    cpt = 1
    while True:
        pt = step * cpt
        if pt + split_size >= size:
            if split_size == size:
                break
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        cpt += 1
    return points

starting_points_X = starting_points(width, 16, 0.5)
starting_points_Y = starting_points(height, 16, 0.5)

splitted_ndg = []
for i in starting_points_Y:
    for j in starting_points_X:
        splitted_ndg = image_ndg[i:i+16, j:j+16]

#Calcul DCT
DCT = np.float32(splitted_ndg)
for i in splitted_ndg:
    DCT = cv2.dct(DCT)

#Tri lexicographique
features = []
DCT = DCT[~np.all(DCT == 0, axis=1)]
features = DCT[np.lexsort(np.rot90(DCT))]

