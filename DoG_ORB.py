from datetime import datetime
from skimage.feature import blob_dog, match_descriptors
from math import sqrt
import cv2
import numpy as np
import scipy
from scipy import ndimage
from scipy.spatial import distance
import glob
import os
import math

orb = cv2.ORB_create(1000)
matcher = cv2.DescriptorMatcher_create(
    cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)


def sobel(image):
    inputImage = image.astype(int)
    dx = ndimage.sobel(inputImage, 1)
    dy = ndimage.sobel(inputImage, 0)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    sobelImage = np.uint8(mag)
    return sobelImage


def dog(greyImage):
    blobs_dog = blob_dog(greyImage, max_sigma=100, threshold=.05)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    return blobs_dog


def show(blobs_all):
    blob_area = []
    blobs_list = [blobs_all]
    for blobs in blobs_list:
        for blob in blobs:
            y, x, r = blob
            area = [y, x, r]
            if 2*r > 1:
                # print area
                blob_area.append(area)
    return blob_area


if __name__=='__main__':
    start_time = datetime.now()
    image = cv2.imread('data/iran.png')
    sobelImage = sobel(image)
    sobelImageGrey = cv2.cvtColor(sobelImage, cv2.COLOR_BGR2GRAY)
    imageGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blobs_all = dog(sobelImageGrey)
    output = show(blobs_all)

    imageClone = image.copy()
    key, des = orb.detectAndCompute(imageGrey, None)

    src = np.array([]).reshape(-1, 1, 2)
    dst = np.array([]).reshape(-1, 1, 2)

    geom = 0

    liste1 = []
    for blob in range(0, len(output)):
        bloby, blobx, blobr = output[blob]
        cv2.circle(imageClone, (int(blobx), int(bloby)), int(blobr), (255, 0, 0), 1)
        liste2 = []
        kp1 = []
        ds1 = []
        liste3 = []
        index = 0

        for k,d in zip(key, des):
            if (k.pt[0] - blobx)**2 + (k.pt[1] - bloby)**2 <= (blobr**2):
                liste2.append(index)
                kp1.append(k)
                ds1.append(d)
            index += 1

        if liste2:
            kp2 = np.delete(key, liste2, axis=0)
            ds2 = np.delete(des, liste2, axis=0)
            nnMatches = matcher.knnMatch(np.array(ds1), ds2, 2)

            goodMatch = []

            nnMatchRatio = 0.6

            for m,n in nnMatches:
                if m.distance < nnMatchRatio * n.distance:
                    goodMatch.append(m)

            
            MIN_MATCH_COUNT = 0
            if len(goodMatch) > MIN_MATCH_COUNT:
                srcPoints = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
                dstPoints = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
                src = np.array(srcPoints).ravel()
                dst = np.array(dstPoints).ravel()

                ps = np.array(src).reshape((-1, 2))
                pd = np.array(dst).reshape((-1, 2))

                for k1, k2 in zip(ps, pd):
                    cv2.circle(imageClone, (int(k1[0]), int(k1[1])), 4, (255, 0, 0), -1)
                    cv2.circle(imageClone, (int(k2[0]), int(k2[1])), 4, (0, 0, 255), -1)
                    cv2.line(imageClone,(int(k1[0]),int(k1[1])),(int(k2[0]),int(k2[1])),(0,255,0),2)   

cv2.imwrite('Resultats/iran_orbdog_detection.png', imageClone)
end_time = datetime.now()
print('Duartion: {}'.format(end_time - start_time))