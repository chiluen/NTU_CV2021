import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

#以原點為center
img = cv2.imread('lena.bmp', 0)

kernel = [[-2,-1], [-2,0], [-2,1], 
          [-1,-2], [-1,-1], [-1,0], [-1,1], [-1,2],
          [0,-2], [0,-1], [0,0], [0,1], [0,2],
          [1,-2], [1,-1], [1,0], [1,1], [1,2],
          [2,-1], [2,0], [2,1]]

# make binary picture
img_binary = np.zeros((512,512))
for row in range(img.shape[0]):
    for col in range(img.shape[1]):
        if img[row][col] > 127:
            img_binary[row][col] = 255
        else:
            img_binary[row][col] = 0

#hw4-1
def dilation(img_binary, kernel):
    print("Process for dilation image...")
    img_dilation = np.zeros((512,512))
    for row in range(img_binary.shape[0]):
        for col in range(img_binary.shape[1]):
            if img_binary[row][col] == 255: #kernel中心點跟img_binary對齊
                for y,x in kernel:
                    if (y+row) >=0 and (y+row) < 512 and (x+col) >= 0 and (x+col) <512:#boundary check
                        img_dilation[y+row][x+col] = 255
    print("Finish")
    return img_dilation

#hw4-2
def erosion(img_binary, kernel):
    print("Process for erosion image...")
    img_erosion = np.zeros((512,512))
    for row in range(img_binary.shape[0]):
        for col in range(img_binary.shape[1]):
            flag_interset = True
            for y,x in kernel:
                if (y+row) <0 or (y+row) >= 512 or (x+col) < 0 or (x+col) >= 512: #boundary check
                    flag_interset = False
                    break
                if img_binary[y+row][x+col] != 255:
                    flag_interset = False
                    break
            if flag_interset:
                img_erosion[row][col] = 255
    print("Finish")
    return img_erosion

#hw4-3
#先做erosion, 再做dilation
def opening(img_binary, kernel):
    print("Process for opening image...")
    return dilation(erosion(img_binary, kernel), kernel)

#hw4-4
#先做dilation, 再做erosion
def closing(img_binary, kernel):
    print("Process for closing image...")
    return erosion(dilation(img_binary, kernel), kernel)


#hw4-5
#hit and miss: L kernel 是foreground, J kernel是backgound
def hitandmiss(img_binary):
    print("Process for hitandmiss image...")
    kernel_J = [[0,-1], [0,0], [1,0]]
    kernel_K = [[-1,0], [-1,1], [0,1]]
    img_c = np.zeros((512,512))
    for row in range(img_binary.shape[0]):
        for col in range(img_binary.shape[1]):
            if img_binary[row][col] == 255:
                img_c[row][col] = 0
            else:
                img_c[row][col] = 255

    img_hit = erosion(img_binary, kernel_J)
    img_miss = erosion(img_c, kernel_K)
    img_hitandmiss = np.zeros((512,512))
    for row in range(img_binary.shape[0]): #intersect
        for col in range(img_binary.shape[1]):
            if img_hit[row][col] == 255 and img_miss[row][col] == 255:
                img_hitandmiss[row][col] = 255
    print("Finish")
    return img_hitandmiss


if __name__ == "__main__":

    img_dilation = dilation(img_binary, kernel)             
    cv2.imwrite('img_dilation.png', img_dilation)

    img_erosion = erosion(img_binary, kernel)
    cv2.imwrite('img_erosion.png', img_erosion)

    img_opening = opening(img_binary, kernel)
    cv2.imwrite('img_opening.png', img_opening)

    img_closing = closing(img_binary, kernel)
    cv2.imwrite('img_closing.png', img_closing)

    img_hitandmiss = hitandmiss(img_binary)
    cv2.imwrite('img_hitandmiss.png', img_hitandmiss)


