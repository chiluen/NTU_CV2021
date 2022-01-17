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

#hw5-1
def dilation(img, kernel):
    print("Process for gray-scale dilation image...")
    img_dilation = np.zeros((512,512))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row][col] > 0: #kernel中心點跟img對齊, 且大於0
                MaxValue=-1
                #找max數值
                for y,x in kernel:
                    if (y+row) >=0 and (y+row) < 512 and (x+col) >= 0 and (x+col) <512:#boundary check
                        if img[y+row][x+col] > MaxValue:
                            MaxValue = img[y+row][x+col]
                for y,x in kernel:
                    if (y+row) >=0 and (y+row) < 512 and (x+col) >= 0 and (x+col) <512:#boundary check
                        img_dilation[y+row][x+col] = MaxValue
    return img_dilation


#hw5-2
def erosion(img, kernel):
    print("Process for erosion image...")
    img_erosion = np.zeros((512,512))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            MinValue = 256
            for y,x in kernel:
                if (y+row) >=0 and (y+row) < 512 and (x+col) >= 0 and (x+col) <512:#boundary check
                    if img[y+row][x+col] < MinValue:
                        MinValue = img[y+row][x+col]
            img_erosion[row][col] = MinValue
    return img_erosion

#hw5-3
#先做erosion, 再做dilation
def opening(img, kernel):
    print("Process for opening image...")
    return dilation(erosion(img, kernel), kernel)

#hw5-4
#先做dilation, 再做erosion
def closing(img, kernel):
    print("Process for closing image...")
    return erosion(dilation(img, kernel), kernel)


if __name__ == "__main__":

    img_dilation = dilation(img, kernel)
    cv2.imwrite('img_dilation.png', img_dilation)

    img_erosion = erosion(img, kernel)
    cv2.imwrite('img_erosion.png', img_erosion)

    img_opening = opening(img, kernel)
    cv2.imwrite('img_opening.png', img_opening)

    img_closing = closing(img, kernel)
    cv2.imwrite('img_closing.png', img_closing)



