import cv2
import numpy as np
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from numpy import dot
from numpy.linalg import norm

forged_img = cv2.imread("images/poissons_falsifiés.png")
grey_forged_img = cv2.cvtColor(forged_img, cv2.COLOR_BGR2GRAY)
red_channel = forged_img[:, :, 2]
green_channel = forged_img[:, :, 1]
blue_channel = forged_img[:, :, 0]
block_size = 15*15
block_width = 15
block_height = 15
height = forged_img.shape[0]
width = forged_img.shape[1]
overlapping = 0.5
number_blocks = 0

#Divide image in overlapping blocks
print("Début de calcul des start point...")
def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            if split_size == size:
                break
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

#1734
X_points = start_points(width, block_width, 0.5)
Y_points = start_points(height, block_height, 0.5)
print(X_points)
print(Y_points)
print("End of starting points...")

red_blocks = []
green_blocks = []
blue_blocks = []

print("Début du split de l'image...")
for i in Y_points:
    for j in X_points:
        split_red = red_channel[i:i+block_height, j:j+block_width]
        red_blocks.append(split_red)
        split_green = green_channel[i:i + block_height, j:j + block_width]
        green_blocks.append(split_green)
        split_blue = blue_channel[i:i + block_height, j:j + block_width]
        blue_blocks.append(split_blue)
        number_blocks += 1
print(split_red)
print(split_green)
print(split_blue)
print("End of split image...")

print("Begin of LBP...")
#LBP
def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass

    return new_value


def get_pixel(img, center, x, y):
    new_value = 0

    try:
        if img[x][y] >= center:
            new_value = 1

    except:
        pass

    return new_value


# Function for calculating LBP
def lbp_calculated_pixel(img, x, y):
    center = img[x][y]

    val_ar = []

    # top_left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))

    # top
    val_ar.append(get_pixel(img, center, x - 1, y))

    # top_right
    val_ar.append(get_pixel(img, center, x - 1, y + 1))

    # right
    val_ar.append(get_pixel(img, center, x, y + 1))

    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))

    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))

    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y - 1))

    # left
    val_ar.append(get_pixel(img, center, x, y - 1))

    # Now, we need to convert binary
    # values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]

    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


block_test_lbp = red_blocks[0]
test_lbp = np.zeros((block_width, block_height), np.uint8)

for i in range(0, block_height):
    for j in range(0, block_width):
        test_lbp[i, j] = lbp_calculated_pixel(block_test_lbp, i, j)

block_red = []
block_green = []
block_blue = []
block_red_lbp = np.zeros((block_width, block_height), np.uint8)
block_green_lbp = np.zeros((block_width, block_height), np.uint8)
block_blue_lbp = np.zeros((block_width, block_height), np.uint8)
blocks_red_lbp = []
blocks_green_lbp = []
blocks_blue_lbp = []

for i in range(0, number_blocks):
    block_red = red_blocks[i]
    block_green = green_blocks[i]
    block_blue = blue_blocks[i]
    for j in range(0, block_height):
        for k in range(0, block_width):
            block_red_lbp[j, k] = lbp_calculated_pixel(block_red, j, k)
            block_green_lbp[j, k] = lbp_calculated_pixel(block_green, j, k)
            block_blue_lbp[j, k] = lbp_calculated_pixel(block_blue, j, k)
    blocks_red_lbp.append(block_red_lbp)
    blocks_green_lbp.append(block_green_lbp)
    blocks_blue_lbp.append(block_blue_lbp)

print(blocks_red_lbp[0])
#print(blocks_green_lbp)
#print(blocks_blue_lbp)

print("End of LBP...")

print("Begin of histograms...")
#Histograms
histo_red = []
histo_green = []
histo_blue = []
red_block_histo = []
green_block_histo = []
blue_block_histo = []


for i in range(0, number_blocks):
    histo_red = cv2.calcHist([blocks_red_lbp[i]], [0], None, [256], [0, 256])
    RBH = np.concatenate(histo_red)
    #RBH = RBH.tolist()
    red_block_histo.append(RBH)

    histo_green = cv2.calcHist([blocks_green_lbp[i]], [0], None, [256], [0, 256])
    GBH = np.concatenate(histo_green)
    #GBH = GBH.tolist()
    green_block_histo.append(GBH)

    histo_blue = cv2.calcHist([blocks_blue_lbp[i]], [0], None, [256], [0, 256])
    BBH = np.concatenate(histo_blue)
    #BBH = BBH.tolist()
    blue_block_histo.append(BBH)

print(red_block_histo)

#0 = non correler / plus on se rapproche de 1, plus c'est correler / 1 = correler
print("\n")
result = cosine_distances(red_block_histo[0].reshape(1, -1), red_block_histo[25].reshape(1, -1))
print(result)


print("End of computing histograms...")


print("Primary candidate selection...")
#Primary candidate selection

print(number_blocks)







#Neighbourhood clustering