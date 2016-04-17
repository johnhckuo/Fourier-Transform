# coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

ma = cv2.imread('chiang.jpg',0) #灰階圖片
bian = cv2.imread('mao.jpg',0) 
plt.subplot(221),plt.imshow(ma,'gray'),plt.title("Chiang")
plt.xticks([]),plt.yticks([])
plt.subplot(222),plt.imshow(bian,'gray'),plt.title("Mao")
plt.xticks([]),plt.yticks([])
#--------------------------------
f1 = np.fft.fft2(ma)
f1shift = np.fft.fftshift(f1)
f1_A = np.abs(f1shift) #振幅
f1_P = np.angle(f1shift) #相位
#--------------------------------
f2 = np.fft.fft2(bian)
f2shift = np.fft.fftshift(f2)
f2_A = np.abs(f2shift) #振幅
f2_P = np.angle(f2shift) #相位
#---圖1的振幅--圖2的相位--------------------
img_new1_f = np.zeros(ma.shape,dtype=complex) 
img1_real = f1_A*np.cos(f2_P) #取實部
img1_imag = f1_A*np.sin(f2_P) #取虚部
img_new1_f.real = np.array(img1_real) 
img_new1_f.imag = np.array(img1_imag) 
f3shift = np.fft.ifftshift(img_new1_f) #傅立葉逆轉換
img_new1 = np.fft.ifft2(f3shift)
#img_new1為複數，尚無法顯示
img_new1 = np.abs(img_new1)
plt.subplot(223),plt.imshow(img_new1,'gray'),plt.title('Chiang_A + Mao_P')
plt.xticks([]),plt.yticks([])
#---圖2的振幅--圖1的相位--------------------
img_new2_f = np.zeros(ma.shape,dtype=complex) 
img2_real = f2_A*np.cos(f1_P) #取實部
img2_imag = f2_A*np.sin(f1_P) #取虚部
img_new2_f.real = np.array(img2_real) 
img_new2_f.imag = np.array(img2_imag) 
f4shift = np.fft.ifftshift(img_new2_f) #傅立葉逆轉換
img_new2 = np.fft.ifft2(f4shift)
#img_new2為複數，尚無法顯示
img_new2 = np.abs(img_new2)
plt.subplot(224),plt.imshow(img_new2,'gray'),plt.title('Chiang_P + Mao_A')
plt.xticks([]),plt.yticks([])
plt.show()
cv2.waitKey(0);
cv2.destroyAllWindows();

