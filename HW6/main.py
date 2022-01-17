import numpy as np
import cv2

img = cv2.imread('lena.bmp', 0)

#binary image
img_binary = np.zeros(img.shape)
for row in range(img.shape[0]):
    for col in range(img.shape[1]):
        if img[row][col] >= 128:
            img_binary[row][col] = 1

#down sample
img_downsample = np.zeros((64,64))
for row in range(img_downsample.shape[0]):
    for col in range(img_downsample.shape[1]):
        img_downsample[row][col] = img_binary[8*row][8*col]
        
#padding
img_pad = np.zeros((66,66))
for row in range(img_downsample.shape[0]):
    for col in range(img_downsample.shape[1]):
        img_pad[row+1][col+1] = img_downsample[row][col]


def cal_score(x0,x1,x2,x3,x4,x5,x6,x7,x8):
    condition = []
    #first dimension
    if x1==x0:
        if x2==x0 and x6==x0:
            condition.append("r")
        else:
            condition.append("q")
    else:
        condition.append("s")
    
    #second dimension
    if x2==x0:
        if x7==x0 and x3==x0:
            condition.append("r")
        else:
            condition.append("q")
    else:
        condition.append("s")
            
    #third dimension
    if x3==x0:
        if x8==x0 and x4==x0:
            condition.append("r")
        else:
            condition.append("q")
    else:
        condition.append("s")
        
    #third dimension
    if x4==x0:
        if x1==x0 and x5==x0:
            condition.append("r")
        else:
            condition.append("q")
    else:
        condition.append("s")
    
    #Check for q and r
    if condition.count("r") == 4:
        return 5
    else:
        return condition.count("q")


img_yokoi = np.zeros((64,64))
for row in range(img_downsample.shape[0]):
    for col in range(img_downsample.shape[1]):
        center_row = row+1
        center_col = col+1
        x0 = img_pad[center_row][center_col]
        x1 = img_pad[center_row][center_col+1]
        x2 = img_pad[center_row-1][center_col]
        x3 = img_pad[center_row][center_col-1]
        x4 = img_pad[center_row+1][center_col]
        x5 = img_pad[center_row+1][center_col+1]
        x6 = img_pad[center_row-1][center_col+1]
        x7 = img_pad[center_row-1][center_col-1]
        x8 = img_pad[center_row+1][center_col-1]
        if x0 != 0:
            score = cal_score(x0,x1,x2,x3,x4,x5,x6,x7,x8)
            img_yokoi[row][col] = score

# do the redirection
with open("yokoi.txt", "w") as f:
    for i in range(img_yokoi.shape[0]):
        string = ""
        for j in range(img_yokoi.shape[1]):

            if img_yokoi[i][j] == 0:
                string += " "
            else:
                string += str(int(img_yokoi[i][j]))
        string += "\n"
        f.write(string)