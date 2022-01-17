import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--part", help="choose homework part from[1,2,3]", type=int, default = 1)
args = parser.parse_args()
if args.part == 1:
    #HW3-1
    img = cv2.imread('lena.bmp', 0) #gray scope
    pixel_list = [0 for num in range(256)]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_list[img[i][j]] += 1

    plt.bar(range(256), pixel_list)
    plt.savefig("hist.png")
if args.part == 2:
#HW3-2
    img = cv2.imread('lena.bmp', 0) #gray scope
    pixel_list = [0 for num in range(256)]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] //= 3
            
            
    cv2.imwrite('img_DivideBy3.jpeg', img)

    pixel_list = [0 for num in range(256)]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_list[img[i][j]] += 1

    plt.bar(range(256), pixel_list)
    plt.savefig("hist_DiviceBy3.png")

if args.part == 3:
    #HW3-3

    img = cv2.imread('img_DivideBy3.jpeg', 0) #gray scope


    #先做出直方圖
    pixel_list = [0 for num in range(256)]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_list[img[i][j]] += 1

    #轉換成機率,乘上最大值並四捨五入,再累加
    sum_temp = 0
    cdf_list = [0 for num in range(256)]
    for i in range(256):
        temp = pixel_list[i]
        temp /= (512*512)
        temp *= 255
        temp = np.round(temp)
        sum_temp += temp
        cdf_list[i] = sum_temp


    #根據list作轉換
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = cdf_list[img[i][j]]

    cv2.imwrite('img_histogram.png', img)


    pixel_list_histogram = [0 for num in range(256)]

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_list_histogram[img[i][j]] += 1

    plt.bar(range(256), pixel_list_histogram)
    plt.savefig("hist_histogram.png")