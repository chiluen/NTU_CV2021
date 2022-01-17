"""
Usage:
Select from [upside_down, rightside_left, diagonally_flip, rotate, shrink, binarize]
"""

import cv2
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", help="choose action", type=str)
args = parser.parse_args()

img = cv2.imread('lena.bmp')


def upside_down(img):
    temp_img = np.zeros((512,512,3))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            temp_img[512 - 1 - row][col] = img[row][col]
    return temp_img
def rightside_left(img):
    temp_img = np.zeros((512,512,3))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            temp_img[row][512 - 1 - col] = img[row][col]
    return temp_img
def diagonally_flip(img):
    temp_img = np.zeros((512,512,3))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            temp_img[col][row] = img[row][col]
    return temp_img
def rotate(img, angle):
    (h, w) = img.shape[:2]
    center = (h/2, w/2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
 
    return rotated

def shrink(img):
    (h, w) = img.shape[:2]
    img_modified = cv2.resize(img, (int(h/2), int(w/2)), interpolation=cv2.INTER_AREA)
    return img_modified

def binarize(img):
    temp_img = np.zeros((512,512,3))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            for channel in range(3):
                if img[row][col][channel] > 127:
                    temp_img[row][col][channel] = 255
                else:
                    temp_img[row][col][channel] = 0
    return temp_img


if args.method == "upside_down":
    img_modified = upside_down(img)
    cv2.imwrite('upside_down.png', img_modified)
    print("Finish")
    
    
elif args.method == "rightside_left":
    img_modified = rightside_left(img)
    cv2.imwrite('rightside_left.png', img_modified)
    print("Finish")
elif args.method == "diagonally_flip":
    img_modified = diagonally_flip(img)
    cv2.imwrite('diagonally_flip.png', img_modified)
    print("Finish")
    
elif args.method == "rotate":
    img_modified = rotate(img, -45)
    cv2.imwrite('rotate_45.png', img_modified)
    print("Finish")
    
elif args.method == "shrink":
    img_modified = shrink(img)
    cv2.imwrite('shrink.png', img_modified)
    print("Finish")
elif args.method == "binarize":
    img_modified = binarize(img)
    cv2.imwrite('binarize.png', img_modified)
    print("Finish")
    
    
    
    
    
    
    
    
    
    