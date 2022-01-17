import numpy as np
import cv2
import random
import copy

img = cv2.imread('lena.bmp', 0)

def Gaussian_noise_10(img):
    img_noise = copy.deepcopy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            noisepixel = int(img[i][j] + 10*random.gauss(0,1))
            if noisepixel > 255:
                noisepixel = 255
            img_noise[i][j] = noisepixel
    return img_noise

def Gaussian_noise_30(img):
    img_noise = copy.deepcopy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            noisepixel = int(img[i][j] + 30*random.gauss(0,1))
            if noisepixel > 255:
                noisepixel = 255
            img_noise[i][j] = noisepixel
    return img_noise

def Saltandpepper_05(img):
    img_noise = copy.deepcopy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            random_value = random.uniform(0,1)
            if random_value <=0.05:
                img_noise[i][j] = 0
            elif random_value >= (1-0.05):
                img_noise[i][j] = 255
            else:
                pass
    return img_noise

def Saltandpepper_10(img):
    img_noise = copy.deepcopy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            random_value = random.uniform(0,1)
            if random_value <=0.1:
                img_noise[i][j] = 0
            elif random_value >= (1-0.1):
                img_noise[i][j] = 255
            else:
                pass
    return img_noise

def box_filter_3(img):
    img_pad = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    img_clean = np.zeros((512,512))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            center_row = row+1
            center_col = col+1
            score = 0
            for i in range(-1,2): #-1,0,1
                for j in range(-1,2):
                    score += img_pad[center_row+i][center_col+j]
            score /= 9
            img_clean[row][col] = score
    return img_clean

def box_filter_5(img):
    img_pad = cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_REPLICATE)
    img_clean = np.zeros((512,512))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            center_row = row+2
            center_col = col+2
            score = 0
            for i in range(-2,3): #-2,-1,0,1,2
                for j in range(-2,3):
                    score += img_pad[center_row+i][center_col+j]
            score /= 25
            img_clean[row][col] = score
    return img_clean

def median_filter_3(img):
    img_pad = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    img_clean = np.zeros((512,512))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            center_row = row+1
            center_col = col+1
            score = []
            for i in range(-1,2): #-1,0,1
                for j in range(-1,2):
                    score.append(img_pad[center_row+i][center_col+j])
            img_clean[row][col] = sorted(score)[4] #median
    return img_clean     

def median_filter_5(img):
    img_pad = cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_REPLICATE)
    img_clean = np.zeros((512,512))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            center_row = row+2
            center_col = col+2
            score = []
            for i in range(-2,3): #-2,-1,0,1,2
                for j in range(-2,3):
                    score.append(img_pad[center_row+i][center_col+j])
            img_clean[row][col] = sorted(score)[12]
    return img_clean

def dilation(img, kernel):
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


def erosion(img, kernel):
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


#先做erosion, 再做dilation
def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)

#先做dilation, 再做erosion
def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)


def SNR(img, img_noise):
    """
    要先做完normalization!
    """
    height = img.shape[0]
    width = img.shape[1]
    img_binary = (img / 255)
    img_noise_binary = (img_noise / 255)

    
    #for original image
    mean = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            mean += img_binary[i][j]
    mean /= (height*width)
    
    var = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            var += (img_binary[i][j] - mean)**2
    var /= (height*width)        
    
    #for noise image
    mean_noise = 0
    for i in range(img_noise.shape[0]):
        for j in range(img_noise.shape[1]):
            mean_noise += (img_noise_binary[i][j] - img_binary[i][j])
    mean_noise /= (height*width)

    var_noise = 0
    for i in range(img_noise.shape[0]):
        for j in range(img_noise.shape[1]):
            var_noise += (img_noise_binary[i][j] - img_binary[i][j] - mean_noise)**2
    var_noise /= (height*width)
    final_score = 20 * np.log10( (var**0.5) / (var_noise**0.5) )
    return final_score

kernel = [[-2,-1], [-2,0], [-2,1], 
          [-1,-2], [-1,-1], [-1,0], [-1,1], [-1,2],
          [0,-2], [0,-1], [0,0], [0,1], [0,2],
          [1,-2], [1,-1], [1,0], [1,1], [1,2],
          [2,-1], [2,0], [2,1]]

img_gaussian_10 = Gaussian_noise_10(img)
cv2.imwrite('img_gaussian_10.png', img_gaussian_10)

img_gaussian_30 = Gaussian_noise_30(img)
cv2.imwrite('img_gaussian_30.png', img_gaussian_30)

img_salt_05 = Saltandpepper_05(img)
cv2.imwrite('img_salt_05.png', img_salt_05)

img_salt_10 = Saltandpepper_10(img)
cv2.imwrite('img_salt_10.png', img_salt_10)

print(SNR(img, img_gaussian_10))
print(SNR(img, img_gaussian_30))
print(SNR(img, img_salt_05))
print(SNR(img, img_salt_10))

#每一個noise都要跑6次operation
#還要做SNR
"""
img_salt_10
"""

c = median_filter_3(img_salt_10)
print(SNR(img, c))
cv2.imwrite('img_salt_10_median_3.png', c)

c = median_filter_5(img_salt_10)
print(SNR(img, c))
cv2.imwrite('img_salt_10_median_5.png', c)

c = box_filter_3(img_salt_10)
print(SNR(img, c))
cv2.imwrite('img_salt_10_box_3.png', c)

c = box_filter_5(img_salt_10)
print(SNR(img, c))
cv2.imwrite('img_salt_10_box_5.png', c)

c = closing(opening(img_salt_10,kernel),kernel)
print(SNR(img, c))
cv2.imwrite('img_salt_10_open.png', c)

c = opening(closing(img_salt_10,kernel),kernel)
print(SNR(img, c))
cv2.imwrite('img_salt_10_close.png', c)

#每一個noise都要跑6次operation
#還要做SNR
"""
img_salt_5
"""

c = median_filter_3(img_salt_05)
print(SNR(img, c))
cv2.imwrite('img_salt_05_median_3.png', c)

c = median_filter_5(img_salt_05)
print(SNR(img, c))
cv2.imwrite('img_salt_05_median_5.png', c)

c = box_filter_3(img_salt_05)
print(SNR(img, c))
cv2.imwrite('img_salt_05_box_3.png', c)

c = box_filter_5(img_salt_05)
print(SNR(img, c))
cv2.imwrite('img_salt_05_box_5.png', c)

c = closing(opening(img_salt_05,kernel),kernel)
print(SNR(img, c))
cv2.imwrite('img_salt_05_open.png', c)

c = opening(closing(img_salt_05,kernel),kernel)
print(SNR(img, c))
cv2.imwrite('img_salt_05_close.png', c)

#每一個noise都要跑6次operation
#還要做SNR
"""
img_gaussian_10
"""

c = median_filter_3(img_gaussian_10)
print(SNR(img, c))
cv2.imwrite('img_gaussian_10_median_3.png', c)

c = median_filter_5(img_gaussian_10)
print(SNR(img, c))
cv2.imwrite('img_gaussian_10_median_5.png', c)

c = box_filter_3(img_gaussian_10)
print(SNR(img, c))
cv2.imwrite('img_gaussian_10_box_3.png', c)

c = box_filter_5(img_gaussian_10)
print(SNR(img, c))
cv2.imwrite('img_gaussian_10_box_5.png', c)

c = closing(opening(img_gaussian_10,kernel),kernel)
print(SNR(img, c))
cv2.imwrite('img_gaussian_10_open.png', c)

c = opening(closing(img_gaussian_10,kernel),kernel)
print(SNR(img, c))
cv2.imwrite('img_gaussian_10_close.png', c)

#每一個noise都要跑6次operation
#還要做SNR
"""
img_gaussian_30
"""

c = median_filter_3(img_gaussian_30)
print(SNR(img, c))
cv2.imwrite('img_gaussian_30_median_3.png', c)

c = median_filter_5(img_gaussian_30)
print(SNR(img, c))
cv2.imwrite('img_gaussian_30_median_5.png', c)

c = box_filter_3(img_gaussian_30)
print(SNR(img, c))
cv2.imwrite('img_gaussian_30_box_3.png', c)

c = box_filter_5(img_gaussian_30)
print(SNR(img, c))
cv2.imwrite('img_gaussian_30_box_5.png', c)

c = closing(opening(img_gaussian_30,kernel),kernel)
print(SNR(img, c))
cv2.imwrite('img_gaussian_30_open.png', c)

c = opening(closing(img_gaussian_30,kernel),kernel)
print(SNR(img, c))
cv2.imwrite('img_gaussian_30_close.png', c)