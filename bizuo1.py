import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import math

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False       # 中文标题设置

img = cv2.imread('GBGWNTest1.jpg')
img = img[:, :, ::-1]

# 高斯滤波函数，滤除高斯噪声
def gaussfilter(img,s=3):           
    b_gray, g_gray, r_gray = cv2.split(img.copy())
    b_blur = cv2.GaussianBlur(b_gray,(s,s),0)
    g_blur = cv2.GaussianBlur(g_gray,(s,s),0)
    r_blur = cv2.GaussianBlur(r_gray,(s,s),0)
    blurred = cv2.merge([b_blur, g_blur, r_blur])
    return blurred,b_blur,g_blur,r_blur


 # 生成高斯模糊核,用于维纳滤波
def fspecial(kernel_size=5,sigma=1.6):         
    m=n=(kernel_size-1.)/2.
    y,x=ogrid[-m:m+1,-n:n+1]
    h=exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < finfo(h.dtype).eps*h.max() ] = 0
    sumh=h.sum()
    if sumh!=0:
        h/=sumh
    return h


def wiener(input, PSF, eps, K=0.3):        # 维纳滤波器，K=0.01
    input_fft = fft.fftshift(fft.fft2(input))
    PSF_fft = fft.fftshift(fft.fft2(PSF,s=[input.shape[0],input.shape[1]]) + eps)
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(fft.ifftshift(input_fft * PSF_fft_1))
    result = np.abs(result)
    return result

def normal(array):
    for i in range(array.shape[0]):
         for j in range(array.shape[1]):
             if    array[i,j]<0: array[i,j] = 0
             elif  array[i,j]>255: array[i,j] = 255
             else: array[i,j] = uint8(array[i,j])
    array = uint8(array)
    return array


def cut(input):                  # 单阈值分割函数
    (h,w) = input.shape[0:2]
    output = []
    ret1,th1 = cv2.threshold(input[:,:,0],215,255,cv2.THRESH_BINARY)
    ret2,th2 = cv2.threshold(input[:,:,1],215,255,cv2.THRESH_BINARY)
    ret3,th3 = cv2.threshold(input[:,:,2],215,255,cv2.THRESH_BINARY)
    output = th1 + th2 + th3   
    return output


blurred,b_blur,g_blur,r_blur = gaussfilter(img, s=3)
PSF = fspecial(kernel_size=5,sigma=1.6)   # 生成5*5高斯滤波器
img0,b0_blur,g0_blur,r0_blur = gaussfilter(img, s=15)
# 维纳滤波   
b_result_wiener = normal(wiener(blurred[:,:,0], PSF, 1e-3)) 
g_result_wiener = normal(wiener(blurred[:,:,1], PSF, 1e-3)) 
r_result_wiener = normal(wiener(blurred[:,:,2], PSF, 1e-3)) 


result_wiener = cv2.merge([b_result_wiener, g_result_wiener, r_result_wiener])
# 直方图均衡化增强

result1 = img.copy()
for i in range(3):
    result1[:,:,i] = cv2.equalizeHist(result_wiener[:,:,i])

    

result_fenge1 = cut(result1)
result2 , b2_blur , g2_blur , r2_blur = gaussfilter(result1, s=9)
result_fenge2 = cut(result2)
result3 , b3_blur , g3_blur , r3_blur = gaussfilter(result2, s=5)
result_fenge3 = cut(result3)




input = result2
img1 = result1.copy()
img2 = input.copy()
gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 5)
edges = cv2.Canny(gray, 30, 80, apertureSize=3)
minLineLength = 100
maxLineGap = 10



lines1 = cv2.HoughLines(edges, 1, np.pi/180, 120)
x, y, z = lines1.shape
print(lines1)
for i in range(0, x):
        theta = lines1[i][0][1]
        print('theta',theta)
        r = lines1[i][0][0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if (theta > 2):
            cv2.line(img1, (x1, y1), (x2, y2), (255, 0, 0), 2)
       




plt.figure(1)
plt.subplot(131)
plt.imshow(img0)
plt.title('原图', fontsize = 18)
plt.subplot(132)
plt.imshow(result_wiener)
plt.title('维纳滤波复原', fontsize = 18)
plt.subplot(133)
plt.imshow(result1)
plt.title('增强结果', fontsize = 18)

plt.figure(2)
plt.subplot(131)
plt.imshow(result_fenge1,cmap='gray')
plt.title('直接分割结果', fontsize = 18)
plt.subplot(132)
plt.imshow(result_fenge2,cmap='gray')
plt.title('先滤波再分割结果', fontsize = 18)
plt.subplot(133)
plt.imshow(result_fenge3,cmap='gray')
plt.title('两次滤波再分割结果', fontsize = 18)


plt.figure(3)
plt.subplot(131)
plt.imshow(input)
plt.title('原图', fontsize = 18)
plt.subplot(132)
plt.imshow(edges, cmap=plt.cm.gray)
plt.title('canny算子检测边缘', fontsize = 18)
plt.subplot(133)
plt.imshow(img1)
plt.title('HoughLines的结果', fontsize = 18)
plt.show()





