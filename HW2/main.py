import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--part", help="choose homework part from[1,2,3]", type=int, default = 1)
args = parser.parse_args()

# hw2-1
if args.part == 1:
    img = cv2.imread('lena.bmp', 0) #gray scope
    img_new = np.zeros(shape=(512,512))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 128:
                img_new[i][j] = 255
            else:
                img_new[i][j] = 0
    cv2.imwrite('binary.png', img_new)

# hw2-2
if args.part == 2:
    img = cv2.imread('lena.bmp', 0) #gray scope
    pixel_list = [0 for num in range(256)]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_list[img[i][j]] += 1

    plt.bar(range(256), pixel_list)
    plt.savefig("hist.png")

# hw2-3
if args.part == 3:

    img = cv2.imread('binary.png', 0)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 0:
                img[i][j] = 1 # background
            else:
                img[i][j] = 0 #forground

    #4連通
    #First pass
    def set_search(index, set_list):
        index = index
        for i in range(len(set_list)):
            for component in set_list[i]:
                if index == component:
                    return i
        return -1

    img_label = np.zeros(shape=(512,512))
    set_list = []
    now_label_index = 1
    counter = 0
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row][col] > 0:
                #check for img_label, and get left/up label
                #0代表沒有
                left_label = 0
                up_label = 0
                
                #check for bound
                if (row-1) >= 0 and (col-1) >= 0: 
                    up_label = img_label[row-1][col]
                    left_label = img_label[row][col-1]
                elif (row-1) <0 and (col-1) >= 0: 
                    left_label = img_label[row][col-1]
                elif (row-1) >=0 and (col-1) < 0:
                    up_label = img_label[row-1][col]
                
                #check for minimum
                """
                建立set_list原則：
                若左/上都是背景:不用建set
                若左/上單邊非背景:不用建set
                若左/上都非背景:
                    查找背景，若兩個都-1，則建立set把兩個包進去
                    查找背景，若一個有值一個-1，則把-1包到有值的
                    查找背景，若兩個都有值，合併兩個set，並且刪除重複的set
                """
                if left_label == 0 and up_label == 0:
                    img_label[row][col] = now_label_index
                    now_label_index += 1
                elif left_label == 0 and up_label != 0:
                    img_label[row][col] = up_label
                elif left_label != 0 and up_label == 0:
                    img_label[row][col] = left_label
                else: #左/上都有值
                    img_label[row][col] = min(left_label, up_label)
                    
                    #加快速度
                    if left_label == up_label:
                        left_set = set_search(left_label,set_list)
                        if left_set == -1:
                            set_list.append({left_label})
                        else:
                            pass
                        continue

                    left_set = set_search(left_label,set_list) #index of set_list, which contain left_label
                    up_set = set_search(up_label,set_list)
                    if left_set == -1 and up_set == -1:
                        set_list.append({left_label, up_label})
                    elif left_set == -1 and up_set != -1:
                        set_list[up_set].add(left_label)
                    elif left_set != -1 and up_set == -1:
                        set_list[left_set].add(up_label)
                    else:
                        if left_set == up_set: #同一個set
                            pass
                        else:
                            #保留left_set, 刪除up_set
                            set_list[left_set] = set_list[left_set].union(set_list[up_set])
                            set_list.pop(up_set)

    #two pass
    dic_count = {}
    for row in range(img_label.shape[0]):
        for col in range(img_label.shape[1]):
            if img_label[row][col] != 0:
                set_index = set_search(img_label[row][col], set_list) #有-1, 代表說有人的分群沒有被set記錄到, 是有可能的
                if set_index == -1: 
                    continue
                img_label[row][col] = min(set_list[set_index])
                try:
                    dic_count[min(set_list[set_index])] += 1
                except:
                    dic_count[min(set_list[set_index])] = 1
    #sorted(dic_count.items(), key=lambda item: item[1], reverse=True) #check for five bounding box

    boundbox_index_list = [k for k,v in dic_count.items() if v > 500]

    def boundbox(boundbox_index, img_label):
        min_x = 512
        min_y = 512
        max_x = 0
        max_y = 0
        for row in range(img_label.shape[0]):
            for col in range(img_label.shape[1]):
                if img_label[row][col] == boundbox_index:
                    if col > max_x:
                        max_x = col
                    if col < min_x:
                        min_x = col
                    if row > max_y:
                        max_y = row
                    if row < min_y:
                        min_y = row
        center_x = min_x + ((max_x - min_x)/2)
        center_y = min_y + ((max_y - min_y)/2)
        return min_x, min_y, max_x, max_y, int(center_x), int(center_y)

    boundbox_result = []
    for boundbox_index in boundbox_index_list:
        boundbox_result.append(boundbox(boundbox_index, img_label))

    img_modified = cv2.imread('lena.bmp')
    for i in range(len(boundbox_result)):

        boundbox_result[i][0] #min_x
        boundbox_result[i][1] #min_y
        boundbox_result[i][2] #max_x
        boundbox_result[i][3] #max_y
        leftup = (boundbox_result[i][0], boundbox_result[i][1])
        rightup = (boundbox_result[i][2], boundbox_result[i][1])
        leftdown = (boundbox_result[i][0], boundbox_result[i][3])
        rightdown = (boundbox_result[i][2], boundbox_result[i][3])

        leftcross = (boundbox_result[i][4]-10,boundbox_result[i][5])
        rightcross = (boundbox_result[i][4]+10,boundbox_result[i][5])
        upcross = (boundbox_result[i][4],boundbox_result[i][5]+10)
        downcross = (boundbox_result[i][4],boundbox_result[i][5]-10)
        
        cv2.line(img_modified, leftup, rightup, (0, 0, 255), 3)
        cv2.line(img_modified, rightup, rightdown, (0, 0, 255), 3)
        cv2.line(img_modified, rightdown, leftdown, (0, 0, 255), 3)
        cv2.line(img_modified, leftdown, leftup, (0, 0, 255), 3)
        cv2.line(img_modified, leftcross, rightcross, (0, 0, 255), 3)
        cv2.line(img_modified, upcross, downcross, (0, 0, 255), 3)
    cv2.imwrite('crossimg.png', img_modified)


