# %%
import numpy as np
import cv2

img = cv2.imread('lena.bmp', 0)

# %%
# Robert operator
"""
先右邊，下面做padding
"""
def robert_operator(img):
    threshold = 30
    img_pad = np.pad(img,((0,1),(0,1)), 'edge').astype('int')
    img_new = np.zeros((512,512))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            x0 = col
            y0 = row
            x1 = x0+1
            y1 = y0+1
            r1 = int(img_pad[y1][x1]) - int(img_pad[y0][x0])
            r2 = img_pad[y1][x0] - img_pad[y0][x1]
            gradient = (r1**2 + r2**2)**(1/2)
            if gradient >= threshold:
                img_new[row][col] = 0
            else:
                img_new[row][col] = 255
    return img_new

def prewitt_operator(img):
    threshold = 24
    img_pad = np.pad(img,((1,1),(1,1)), 'edge').astype('int')
    img_new = np.zeros((512,512))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            x_center = col+1
            x_back = x_center-1
            x_front = x_center+1
            y_center = row+1
            y_back = y_center-1
            y_front = y_center+1
            p1 = (img_pad[y_front][x_back] + img_pad[y_front][x_center] + img_pad[y_front][x_front]) - (img_pad[y_back][x_back] + img_pad[y_back][x_center] + img_pad[y_back][x_front])
            p2 = (img_pad[y_back][x_front] + img_pad[y_center][x_front] + img_pad[y_front][x_front]) - (img_pad[y_back][x_back] + img_pad[y_center][x_back] + img_pad[y_front][x_back])
            gradient = (p1**2 + p2**2)**(1/2)
            if gradient >= threshold:
                img_new[row][col] = 0
            else:
                img_new[row][col] = 255
    return img_new

def sobel_operator(img):
    threshold = 38
    img_pad = np.pad(img,((1,1),(1,1)), 'edge').astype('int')
    img_new = np.zeros((512,512))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            x_center = col+1
            x_back = x_center-1
            x_front = x_center+1
            y_center = row+1
            y_back = y_center-1
            y_front = y_center+1
            s1 = (img_pad[y_front][x_back] + 2*img_pad[y_front][x_center] + img_pad[y_front][x_front]) - (img_pad[y_back][x_back] + 2*img_pad[y_back][x_center] + img_pad[y_back][x_front])
            s2 = (img_pad[y_back][x_front] + 2*img_pad[y_center][x_front] + img_pad[y_front][x_front]) - (img_pad[y_back][x_back] + 2*img_pad[y_center][x_back] + img_pad[y_front][x_back])
            gradient = (s1**2 + s2**2)**(1/2)
            if gradient >= threshold:
                img_new[row][col] = 0
            else:
                img_new[row][col] = 255
    return img_new

def FreiChen_operator(img):
    threshold = 30
    img_pad = np.pad(img,((1,1),(1,1)), 'edge').astype('int')
    img_new = np.zeros((512,512))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            x_center = col+1
            x_back = x_center-1
            x_front = x_center+1
            y_center = row+1
            y_back = y_center-1
            y_front = y_center+1
            s1 = (img_pad[y_front][x_back] + (2**(1/2))*img_pad[y_front][x_center] + img_pad[y_front][x_front]) - (img_pad[y_back][x_back] + (2**(1/2))*img_pad[y_back][x_center] + img_pad[y_back][x_front])
            s2 = (img_pad[y_back][x_front] + (2**(1/2))*img_pad[y_center][x_front] + img_pad[y_front][x_front]) - (img_pad[y_back][x_back] + (2**(1/2))*img_pad[y_center][x_back] + img_pad[y_front][x_back])
            gradient = (s1**2 + s2**2)**(1/2)
            if gradient >= threshold:
                img_new[row][col] = 0
            else:
                img_new[row][col] = 255
    return img_new

def Kirsch_operator(img):
    threshold = 135
    img_pad = np.pad(img,((1,1),(1,1)), 'edge').astype('int')
    img_new = np.zeros((512,512))
    
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            x_center = col+1
            x_back = x_center-1
            x_front = x_center+1
            y_center = row+1
            y_back = y_center-1
            y_front = y_center+1
            k0 = -3*img_pad[y_back][x_back] + -3*img_pad[y_back][x_center] + 5*img_pad[y_back][x_front] \
                 + -3*img_pad[y_center][x_back] + 5*img_pad[y_center][x_front] \
                 + -3*img_pad[y_front][x_back] + -3*img_pad[y_front][x_center] + 5*img_pad[y_front][x_front]
            k1 = -3*img_pad[y_back][x_back] + 5*img_pad[y_back][x_center] + 5*img_pad[y_back][x_front] \
                 + -3*img_pad[y_center][x_back] + 5*img_pad[y_center][x_front] \
                 + -3*img_pad[y_front][x_back] + -3*img_pad[y_front][x_center] + -3*img_pad[y_front][x_front]
            k2 = 5*img_pad[y_back][x_back] + 5*img_pad[y_back][x_center] + 5*img_pad[y_back][x_front] \
                 + -3*img_pad[y_center][x_back] + -3*img_pad[y_center][x_front] \
                 + -3*img_pad[y_front][x_back] + -3*img_pad[y_front][x_center] + -3*img_pad[y_front][x_front]
            k3 = 5*img_pad[y_back][x_back] + 5*img_pad[y_back][x_center] + -3*img_pad[y_back][x_front] \
                 + 5*img_pad[y_center][x_back] + -3*img_pad[y_center][x_front] \
                 + -3*img_pad[y_front][x_back] + -3*img_pad[y_front][x_center] + -3*img_pad[y_front][x_front]
            k4 = 5*img_pad[y_back][x_back] + -3*img_pad[y_back][x_center] + -3*img_pad[y_back][x_front] \
                 + 5*img_pad[y_center][x_back] + -3*img_pad[y_center][x_front] \
                 + 5*img_pad[y_front][x_back] + -3*img_pad[y_front][x_center] + -3*img_pad[y_front][x_front]
            k5 = -3*img_pad[y_back][x_back] + -3*img_pad[y_back][x_center] + -3*img_pad[y_back][x_front] \
                 + 5*img_pad[y_center][x_back] + -3*img_pad[y_center][x_front] \
                 + 5*img_pad[y_front][x_back] + 5*img_pad[y_front][x_center] + -3*img_pad[y_front][x_front]
            k6 = -3*img_pad[y_back][x_back] + -3*img_pad[y_back][x_center] + -3*img_pad[y_back][x_front] \
                 + -3*img_pad[y_center][x_back] + -3*img_pad[y_center][x_front] \
                 + 5*img_pad[y_front][x_back] + 5*img_pad[y_front][x_center] + 5*img_pad[y_front][x_front]
            k7 = -3*img_pad[y_back][x_back] + -3*img_pad[y_back][x_center] + -3*img_pad[y_back][x_front] \
                 + -3*img_pad[y_center][x_back] + 5*img_pad[y_center][x_front] \
                 + -3*img_pad[y_front][x_back] + 5*img_pad[y_front][x_center] + 5*img_pad[y_front][x_front]
            gradient = np.max([k0,k1,k2,k3,k4,k5,k6,k7])
            if gradient >= threshold:
                img_new[row][col] = 0
            else:
                img_new[row][col] = 255
    return img_new


def Robinson_operator(img):
    threshold = 43
    img_pad = np.pad(img,((1,1),(1,1)), 'edge').astype('int')
    img_new = np.zeros((512,512))
    
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            x_center = col+1
            x_back = x_center-1
            x_front = x_center+1
            y_center = row+1
            y_back = y_center-1
            y_front = y_center+1
            k0 = -1*img_pad[y_back][x_back]  + 1*img_pad[y_back][x_front] \
                 + -2*img_pad[y_center][x_back] + 2*img_pad[y_center][x_front] \
                 + -1*img_pad[y_front][x_back] + 1*img_pad[y_front][x_front]
            
            k1 = 1*img_pad[y_back][x_center] + 2*img_pad[y_back][x_front] \
                 + -1*img_pad[y_center][x_back] + 1*img_pad[y_center][x_front] \
                 + -2*img_pad[y_front][x_back] + -1*img_pad[y_front][x_center] 
            
            k2 = 1*img_pad[y_back][x_back] + 2*img_pad[y_back][x_center] + 1*img_pad[y_back][x_front] \
                 + -1*img_pad[y_front][x_back] + -2*img_pad[y_front][x_center] + -1*img_pad[y_front][x_front]
            
            k3 = 2*img_pad[y_back][x_back] + 1*img_pad[y_back][x_center] \
                 + 1*img_pad[y_center][x_back] + -1*img_pad[y_center][x_front] \
                 + -1*img_pad[y_front][x_center] + -2*img_pad[y_front][x_front]
            
            k4 = 1*img_pad[y_back][x_back] + -1*img_pad[y_back][x_front] \
                 + 2*img_pad[y_center][x_back] + -2*img_pad[y_center][x_front] \
                 + 1*img_pad[y_front][x_back] + -1*img_pad[y_front][x_front]
            
            k5 = -1*img_pad[y_back][x_center] + -2*img_pad[y_back][x_front] \
                 + 1*img_pad[y_center][x_back] + -1*img_pad[y_center][x_front] \
                 + 2*img_pad[y_front][x_back] + 1*img_pad[y_front][x_center] 
            
            k6 = -1*img_pad[y_back][x_back] + -2*img_pad[y_back][x_center] + -1*img_pad[y_back][x_front] \
                 + 1*img_pad[y_front][x_back] + 2*img_pad[y_front][x_center] + 1*img_pad[y_front][x_front]
            k7 = -2*img_pad[y_back][x_back] + -1*img_pad[y_back][x_center]  \
                 + -1*img_pad[y_center][x_back] + 1*img_pad[y_center][x_front] \
                 + 1*img_pad[y_front][x_center] + 2*img_pad[y_front][x_front]
            gradient = np.max([k0,k1,k2,k3,k4,k5,k6,k7])
            if gradient >= threshold:
                img_new[row][col] = 0
            else:
                img_new[row][col] = 255
    return img_new
            
def Nevatia_operator(img):
    threshold = 12500
    img_pad = np.pad(img,((2,2),(2,2)), 'edge').astype('int')
    img_new = np.zeros((512,512))
    
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            x_center = col+2
            x_back = x_center-1
            x_back_two = x_center-2
            x_front = x_center+1
            x_front_two = x_center + 2
            
            y_center = row+2
            y_back = y_center-1
            y_back_two = y_center-2
            y_front = y_center+1
            y_front_two = y_center+2
            
            k0 = 100*img_pad[y_back_two][x_back_two]+100*img_pad[y_back_two][x_back]+100*img_pad[y_back_two][x_center]+100*img_pad[y_back_two][x_front]+100*img_pad[y_back_two][x_front_two] \
                 +100*img_pad[y_back][x_back_two]+100*img_pad[y_back][x_back]+100*img_pad[y_back][x_center]+100*img_pad[y_back][x_front]+100*img_pad[y_back][x_front_two] \
                 + -100*img_pad[y_front][x_back_two]+ -100*img_pad[y_front][x_back]+ -100*img_pad[y_front][x_center]+ -100*img_pad[y_front][x_front]+ -100*img_pad[y_front][x_front_two] \
                 + -100*img_pad[y_front_two][x_back_two]+ -100*img_pad[y_front_two][x_back]+ -100*img_pad[y_front_two][x_center]+ -100*img_pad[y_front_two][x_front]+ -100*img_pad[y_front_two][x_front_two] 
            
            
            k1 = 100*img_pad[y_back_two][x_back_two]+100*img_pad[y_back_two][x_back]+100*img_pad[y_back_two][x_center]+100*img_pad[y_back_two][x_front]+100*img_pad[y_back_two][x_front_two] \
                 +100*img_pad[y_back][x_back_two]+100*img_pad[y_back][x_back]+100*img_pad[y_back][x_center]+78*img_pad[y_back][x_front]+ -32*img_pad[y_back][x_front_two] \
                 +100*img_pad[y_center][x_back_two]+92*img_pad[y_center][x_back]+ -92*img_pad[y_center][x_front]+ -100*img_pad[y_center][x_front_two] \
                 + 32*img_pad[y_front][x_back_two]+ -78*img_pad[y_front][x_back]+ -100*img_pad[y_front][x_center]+ -100*img_pad[y_front][x_front]+ -100*img_pad[y_front][x_front_two] \
                 + -100*img_pad[y_front_two][x_back_two]+ -100*img_pad[y_front_two][x_back]+ -100*img_pad[y_front_two][x_center]+ -100*img_pad[y_front_two][x_front]+ -100*img_pad[y_front_two][x_front_two] 
            
            k2 = 100*img_pad[y_back_two][x_back_two]+100*img_pad[y_back_two][x_back]+100*img_pad[y_back_two][x_center]+32*img_pad[y_back_two][x_front]+ -100*img_pad[y_back_two][x_front_two] \
                 +100*img_pad[y_back][x_back_two]+100*img_pad[y_back][x_back]+92*img_pad[y_back][x_center]+ -78*img_pad[y_back][x_front]+ -100*img_pad[y_back][x_front_two] \
                 +100*img_pad[y_center][x_back_two]+100*img_pad[y_center][x_back]+ -100*img_pad[y_center][x_front]+ -100*img_pad[y_center][x_front_two] \
                 + 100*img_pad[y_front][x_back_two]+ 78*img_pad[y_front][x_back]+ -92*img_pad[y_front][x_center]+ -100*img_pad[y_front][x_front]+ -100*img_pad[y_front][x_front_two] \
                 + 100*img_pad[y_front_two][x_back_two]+ -32*img_pad[y_front_two][x_back]+ -100*img_pad[y_front_two][x_center]+ -100*img_pad[y_front_two][x_front]+ -100*img_pad[y_front_two][x_front_two] 
            
            k3 = -100*img_pad[y_back_two][x_back_two]+ -100*img_pad[y_back_two][x_back]+ 100*img_pad[y_back_two][x_front]+100*img_pad[y_back_two][x_front_two] \
                 + -100*img_pad[y_back][x_back_two]+ -100*img_pad[y_back][x_back]+ 100*img_pad[y_back][x_front]+ 100*img_pad[y_back][x_front_two] \
                 + -100*img_pad[y_center][x_back_two]+ -100*img_pad[y_center][x_back]+ 100*img_pad[y_center][x_front]+ 100*img_pad[y_center][x_front_two] \
                 + -100*img_pad[y_front][x_back_two]+ -100*img_pad[y_front][x_back]+ 100*img_pad[y_front][x_front]+ 100*img_pad[y_front][x_front_two] \
                 + -100*img_pad[y_front_two][x_back_two]+ -100*img_pad[y_front_two][x_back]+ 100*img_pad[y_front_two][x_front]+ 100*img_pad[y_front_two][x_front_two]
            
            k4 = -100*img_pad[y_back_two][x_back_two]+32*img_pad[y_back_two][x_back]+100*img_pad[y_back_two][x_center]+100*img_pad[y_back_two][x_front]+100*img_pad[y_back_two][x_front_two] \
                 + -100*img_pad[y_back][x_back_two]+ -78*img_pad[y_back][x_back]+ 92*img_pad[y_back][x_center]+ 100*img_pad[y_back][x_front]+ 100*img_pad[y_back][x_front_two] \
                 + -100*img_pad[y_center][x_back_two]+ -100*img_pad[y_center][x_back]+ 100*img_pad[y_center][x_front]+ 100*img_pad[y_center][x_front_two] \
                 + -100*img_pad[y_front][x_back_two]+ -100*img_pad[y_front][x_back]+ -92*img_pad[y_front][x_center]+ 78*img_pad[y_front][x_front]+ 100*img_pad[y_front][x_front_two] \
                 + -100*img_pad[y_front_two][x_back_two]+ -100*img_pad[y_front_two][x_back]+ -100*img_pad[y_front_two][x_center]+ -32*img_pad[y_front_two][x_front]+ 100*img_pad[y_front_two][x_front_two]
            
            k5 = 100*img_pad[y_back_two][x_back_two]+100*img_pad[y_back_two][x_back]+100*img_pad[y_back_two][x_center]+100*img_pad[y_back_two][x_front]+100*img_pad[y_back_two][x_front_two] \
                 + -32*img_pad[y_back][x_back_two]+ 78*img_pad[y_back][x_back]+100*img_pad[y_back][x_center]+ 100*img_pad[y_back][x_front]+ 100*img_pad[y_back][x_front_two] \
                 + -100*img_pad[y_center][x_back_two]+ -92*img_pad[y_center][x_back]+ 92*img_pad[y_center][x_front]+ 100*img_pad[y_center][x_front_two] \
                 + -100*img_pad[y_front][x_back_two]+ -100*img_pad[y_front][x_back]+ -100*img_pad[y_front][x_center]+ -78*img_pad[y_front][x_front]+ 32*img_pad[y_front][x_front_two] \
                 + -100*img_pad[y_front_two][x_back_two]+ -100*img_pad[y_front_two][x_back]+ -100*img_pad[y_front_two][x_center]+ -100*img_pad[y_front_two][x_front]+ -100*img_pad[y_front_two][x_front_two]
            

            

            gradient = np.max([k0,k1,k2,k3,k4,k5])
            if gradient >= threshold:
                img_new[row][col] = 0
            else:
                img_new[row][col] = 255
    return img_new
            
            


# %%


# %%
robert_image = robert_operator(img)
cv2.imwrite('robert.png', robert_image)

prewitt_image = prewitt_operator(img)
cv2.imwrite('prewitt.png', prewitt_image)

sobel_image = sobel_operator(img)
cv2.imwrite('sobel.png', sobel_image)

FreiChen_image = FreiChen_operator(img)
cv2.imwrite('FreiChen.png', FreiChen_image)

Kirsch_image = Kirsch_operator(img)
cv2.imwrite('Kirsch.png', Kirsch_image)

Robinson_image = Robinson_operator(img)
cv2.imwrite('Robinson.png', Robinson_image)

Nevatia_image = Nevatia_operator(img)
cv2.imwrite('Nevatia.png', Nevatia_image)


