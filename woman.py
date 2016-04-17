# coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Woman
woman = cv2.imread('woman.png',0) #灰階圖片
square = cv2.imread('square.png',0) 
plt.subplot(331),plt.imshow(woman,'gray'),plt.title('Woman')
plt.xticks([]),plt.yticks([])

#取其振幅及相位
f1 = np.fft.fft2(woman)
f1shift = np.fft.fftshift(f1) #將低頻部份移到中間，以方便觀察
f1_A = np.abs(f1shift) #振幅
f1_P = np.angle(f1shift) #相位

#顯示其振幅和相位
plt.subplot(332),plt.imshow(f1_A,'gray'),plt.title('Woman_Magnitude')
plt.subplot(333),plt.imshow(f1_P,'gray'),plt.title('Woman_Phase')

#Square
plt.subplot(334),plt.imshow(square,'gray'),plt.title('Square')
plt.xticks([]),plt.yticks([])

#取其振幅及相位
f2 = np.fft.fft2(square)
f2shift = np.fft.fftshift(f2)  #將低頻部份移到中間，以方便觀察
f2_A = np.abs(f2shift) #振幅
f2_P = np.angle(f2shift) #相位

#顯示其振幅和相位
plt.subplot(335),plt.imshow(f2_A,'gray'),plt.title('Square_Magnitude')
plt.subplot(336),plt.imshow(f2_P,'gray'),plt.title('Square_Phase')

#---Woman的振幅--Square的相位--------------------
img_new1_f = np.zeros(woman.shape,dtype=complex) 
img1_real = f1_A*np.cos(f2_P) #取實部
img1_imag = f1_A*np.sin(f2_P) #取虚部
img_new1_f.real = np.array(img1_real) 
img_new1_f.imag = np.array(img1_imag) 
f3shift = np.fft.ifftshift(img_new1_f) #傅立葉逆轉換
img_new1 = np.fft.ifft2(f3shift)

#輸出結果img_new1為複數，尚無法顯示
img_new1 = np.abs(img_new1)
plt.subplot(337),plt.imshow(img_new1,'gray'),plt.title('Woman_A + Square_P')
plt.xticks([]),plt.yticks([])

#---Square的振幅--Woman的相位--------------------
img_new2_f = np.zeros(woman.shape,dtype=complex) 
img2_real = f2_A*np.cos(f1_P) #取實部
img2_imag = f2_A*np.sin(f1_P) #取虚部
img_new2_f.real = np.array(img2_real) 
img_new2_f.imag = np.array(img2_imag) 
f4shift = np.fft.ifftshift(img_new2_f) #傅立葉逆轉換
img_new2 = np.fft.ifft2(f4shift)

#輸出結果img_new2為複數，尚無法顯示
img_new2 = np.abs(img_new2)
plt.subplot(338),plt.imshow(img_new2,'gray'),plt.title('Woman_P + Square_A')
plt.xticks([]),plt.yticks([])
plt.show()
cv2.waitKey(0);
cv2.destroyAllWindows();

