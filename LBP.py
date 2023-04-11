import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

#Passage de l'image en nuances de gris
suspicious_image = cv2.imread('data/zebres_forgery.png')
ndg_suspicious_image = cv2.cvtColor(suspicious_image, cv2.COLOR_BGR2GRAY)

block_size = (64, 64)
overlap_size = (32, 32)

num_rows = int(np.ceil((ndg_suspicious_image.shape[0] - block_size[1])/overlap_size[1])) + 1
num_cols = int(np.ceil((ndg_suspicious_image.shape[1] - block_size[0])/overlap_size[0])) + 1

blocks = np.zeros((num_rows, num_cols, block_size[1], block_size[0]), dtype=np.uint8)
block_keypoints = []
block_descriptors = []
LBP_features = []

sift = cv2.SIFT_create()

'''
keypoints, descriptors = sift.detectAndCompute(ndg_suspicious_image, None)

# Perform k-means clustering on the descriptors
kmeans = KMeans(n_clusters=10)
kmeans.fit(descriptors)

# Get cluster assignments for each descriptor
labels = kmeans.predict(descriptors)

# Create a dictionary to store keypoints for each cluster
clusters = {}
for i, label in enumerate(labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(keypoints[i])

# Visualize the clustered keypoints
vis = cv2.drawKeypoints(ndg_suspicious_image, sum(clusters.values(), []), None, color=(0, 255, 0))
cv2.imshow("Clustered Keypoints", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

def get_lbp_features(img_block, radius=1, n_points=8):
    height, width = img_block.shape
    lbp = np.zeros((height, width), dtype=np.uint8)
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            center = img_block[y, x]
            pixel_values = []
            for i in range(n_points):
                angle = float(i) * (2.0 * np.pi / n_points)
                x_i = x + int(round(radius * np.cos(angle)))
                y_i = y - int(round(radius * np.sin(angle)))
                pixel_values.append(img_block[y_i, x_i])
            binary_values = [int(pv >= center) for pv in pixel_values]
            lbp_value = sum([2**i * bv for i, bv in enumerate(binary_values)])
            lbp[y - radius, x - radius] = lbp_value
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 2**n_points + 1), range=(0, 2**n_points))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def block_matching(lbp_features, threshold):
    """
    Identifie les blocs de texture similaires en comparant les histogrammes LBP de chaque paire de blocs.
    
    Args:
        lbp_features: tableau numpy de taille (num_blocks, num_bins), contenant les histogrammes LBP de chaque bloc.
        threshold: seuil de similarité pour considérer deux blocs comme étant similaires. La valeur par défaut est 0.2.
    
    Returns:
        liste de tuples contenant les indices des blocs similaires.
    """
    num_blocks = lbp_features.shape[0]
    similar_blocks = []
    
    # Calcul de la distance euclidienne entre les histogrammes LBP de chaque paire de blocs
    for i in range(num_blocks):
        for j in range(i+1, num_blocks):
            dist = euclidean(lbp_features[i], lbp_features[j])
            # Si la distance est inférieure au seuil, les deux blocs sont considérés comme similaires
            if dist < threshold:
                similar_blocks.append((i, j))
    
    return similar_blocks

def geometric_transform(block_coords, matches):
    # Extract coordinates of matched keypoints in the two blocks
    src_pts = []
    dst_pts = []
    for m in matches:
        if m.queryIdx < len(block_coords) and m.trainIdx < len(block_coords):
            src_pts.append(block_coords[m.queryIdx])
            dst_pts.append(block_coords[m.trainIdx])
    src_pts = np.float32(src_pts).reshape(-1,1,2)
    dst_pts = np.float32(dst_pts).reshape(-1,1,2)

    M = np.zeros((3,3))
    # Compute the transformation matrix using RANSAC
    if len(matches) >= 4:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    block_coords = np.array(block_coords)
    M = M.astype(np.float32)
    # Apply the transformation to the detected block coordinates
    transformed_block_coords = cv2.perspectiveTransform(block_coords.reshape(-1, 1, 2), M)

    return transformed_block_coords

def get_block_desc(block_index, keypoints_list):
    return keypoints_list[block_index]

def match_keypoints(des1, des2):
    bf = cv2.BFMatcher()
    
    # Récupération des descripteurs de deux ensembles de keypoints kp1 et kp2
    des1 = np.array(des1).astype(np.float32)
    des2 = np.array(des2).astype(np.float32)

    # Application de la méthode match() pour trouver les correspondances entre les descripteurs
    matches = bf.match(des1, des2)

    return matches





for i in range(num_rows):
    for j in range(num_cols):
        top = i*overlap_size[1]
        left = j*overlap_size[0]

        bottom = min(top + block_size[1], ndg_suspicious_image.shape[0])
        right = min(left + block_size[0], ndg_suspicious_image.shape[1])

        block = ndg_suspicious_image[top:bottom, left:right]

        hist = get_lbp_features(block, 1, 8)
        LBP_features.append(hist)

        keypoints, descriptors = sift.detectAndCompute(block, None)
        block_keypoints.append(keypoints)
        block_descriptors.append(descriptors)

        #cv2.rectangle(ndg_suspicious_image, (left, top), (right, bottom), 128, 2)

        blocks[i, j, :bottom-top, :right-left] = ndg_suspicious_image[top:bottom, left:right]

#plt.bar(range(len(LBP_features[1])), LBP_features[1])
#plt.savefig('data/histogramme_block_1.png')

LBP_features_array = np.array(LBP_features)
similar_blocks = block_matching(LBP_features_array, 0.02)

for blocks in similar_blocks:
    block_coords = blocks
    des1 = get_block_desc(block_coords[0], block_descriptors)
    des2 = get_block_desc(block_coords[1], block_descriptors)
    matched_keypoints = match_keypoints(des1, des2)
    transformed_block_coords = geometric_transform(block_coords, matched_keypoints)


'''
for kp in keypoints:
    x, y = kp.pt
    x += left
    y += top
    cv2.circle(ndg_suspicious_image, (int(x), int(y)), 2, (0, 0, 255), -1)

cv2.imshow('Grid Image with Keypoints', ndg_suspicious_image)
cv2.waitKey(0)
'''
