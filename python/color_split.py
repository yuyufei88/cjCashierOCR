# -*- coding:utf-8 -*-
# created by zhutao chu

import cv2
import os
import numpy as np

color_range = 40


def getbg(img):
    white = []
    black = []
    img_w = img.shape[1]
    img_h = img.shape[0]
    for i in range(img_w):
        len_x = len(np.where(img[:, i]  == 255)[0])
        white.append(len_x)
        black.append(img_h-len_x)
    white_max = np.max(white)
    black_max = np.max(black)
    if (len(np.where(black  == black_max)[0])>=len(np.where(white  == white_max)[0])):
        return True
    else:
        return False



def getrow(img):
    row_num = []
    img_w = img.shape[1]
    for i in range(img_w):
        row_num.append(len(np.where(img[:,i]==0)[0]))
    return row_num

def getcol(img):
    col_num = []
    img_h = img.shape[0]
    for i in range(img_h):
        col_num.append(len(np.where(img[i,:]==0)[0]))
    return col_num

def getpoint(row_num,col_num):
    left = []
    start = []
    i =1
    while(i<len(row_num)):
        if row_num[i] > row_num[i - 1]:
            start.append(row_num[i - 1])
            j = i
            while(j<len(row_num)-1):
                if row_num[j] <= start[0]:
                    left.append([i-1,j+1])
                    i = j
                    break
                j +=1
        i+=1
    for i in range(1,len(col_num)):
        if col_num[i]>col_num[i-1]:
            top = i-1
            break
    for i in range(len(row_num)-1,1,-1):
        if row_num[i]<row_num[i-1]:
            right = i-1
            break
    for i in range(len(col_num)-1,1,-1):
        if col_num[i]<col_num[i-1]:
            bottom = i
            break
    return left,top,right,bottom

def onmouse(event, x, y, flags, param):   #标准鼠标交互函数
    global b, g, r
    if event==cv2.EVENT_MOUSEMOVE:      #当鼠标移动时
        b,g,r = img[y,x,:]

def color_up(color,color_range):
    colorup = color + color_range // 2
    if colorup > 255:
        colorup = 255
    return colorup

def color_down(color,color_range):
    colordown = color - color_range // 2
    if colordown < 0:
        colordown = 0
    return colordown


def update(x):
    # 回调函数 更新value的值
    color_range = cv2.getTrackbarPos('range','mask')
    sum = cv2.getTrackbarPos('sum','mask')
    b_up = color_up(b, color_range)
    b_down = color_down(b, color_range)
    g_up = color_up(g, color_range)
    g_down = color_down(g, color_range)
    r_up = color_up(r, color_range)
    r_down = color_down(r, color_range)
    color = [
        ([b_down, g_down, r_down], [b_up, g_up, r_up])  # 黄色范围~这个是我自己试验的范围，可根据实际情况自行调整~注意：数值按[b,g,r]排布
    ]
    # 如果color中定义了几种颜色区间，都可以分割出来
    for (lower, upper) in color:
        # 创建NumPy数组
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限

        # 根据阈值找到对应颜色
        mask = cv2.inRange(img, lower, upper)
        dst = mask
    if sum>=1:
        hline = cv2.getStructuringElement(cv2.MORPH_RECT,(sum,sum))
        vline = cv2.getStructuringElement(cv2.MORPH_RECT,(sum,sum))
        dst = cv2.morphologyEx(dst,cv2.MORPH_RECT,hline)
        dst = cv2.morphologyEx(dst,cv2.MORPH_RECT,vline)
    bg_judge = getbg(dst)
    if bg_judge:
        dst = cv2.bitwise_not(dst)
    row_num = getrow(dst)
    col_num = getcol(dst)
    left, top, right, bottom = getpoint(row_num, col_num)
    img_mask = cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)
    img_compose = np.ones([bottom-top,left[len(left)-1][1]-left[0][0]+(len(left))*30,3],np.uint8)*128
    # img_mask = cv2.rectangle(img_mask,(left,top),(right,bottom),(0,0,255))
    print(len(left))
    x0 = 0

    if len(left)<10:
        for i in range(len(left)):
            img_split = img_mask[top:bottom,left[i][0]:left[i][1]]
            x1 = left[i][1]-left[i][0]
            img_compose[0:bottom-top,x0:x0+x1]=img_mask[top:bottom,left[i][0]:left[i][1]]
            x0 = x0 +x1+30
            # cv2.namedWindow(str(i), 0)
            # cv2.imshow(str(i), img_split)
    cv2.imshow("mask", dst)
    cv2.imshow('img_compose', img_compose)
    # output = cv2.bitwise_and(img, img, mask=mask)
    # b_channel, g_channel, r_channel = cv2.split(output)
    # img_BGRA = cv2.merge((b_channel, g_channel, r_channel, mask))
    # img_BGRA = cv2.cvtColor(img_BGRA,cv2.COLOR_BGRA2RGBA)
    # cv2.imwrite('1.png',img_BGRA)


dirname = './test_sum1'
dirpath = os.listdir(dirname)
dirpath.sort()
for file in dirpath:
    img_path = os.path.join(dirname,file)
    img = cv2.imread(img_path)
    cv2.namedWindow('img',0)
    points = cv2.setMouseCallback("img", onmouse)  # 回调绑定窗口
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.namedWindow('mask', 0)
    cv2.createTrackbar('range', 'mask', 0, 255, update)
    cv2.createTrackbar('sum', 'mask', 0, 10, update)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

