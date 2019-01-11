#!/usr/bin/env python
# coding: utf-8


import os
dd=os.listdir('TIN')
f1 = open('train.txt', 'w')
f2 = open('test.txt', 'w')
for i in range(len(dd)):
    d2 = os.listdir ('TIN/%s/images/'%(dd[i]))
    # ~ print(d2)
    for j in range(len(d2)-2):
        str1='TIN/%s/images/%s'%(dd[i], d2[j])
        f1.write("%s %d\n" % (str1, i))
    str1='TIN/%s/images/%s'%(dd[i], d2[-1])
    f2.write("%s %d\n" % (str1, i))

f1.close()
f2.close()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from numpy import linalg as LA
from skimage import data, exposure, img_as_float
from PIL import Image
from PIL import ImageEnhance
import cv2
def custom_blur_demo(image):
    # ~ kernel = np.array([[0, 1, 0],
                       # ~ [1, -4, 1],
                       # ~ [0, 1, 0]], np.float32) #銳利化
    kernel2 = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], np.float32)/9 #模糊化
    # ~ dst = cv2.filter2D(image, -1, kernel)
    dst = cv2.filter2D(image, -1, kernel2)
    #cv2.imshow("custom_blur_demo", dst)
    return dst
  
def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)

    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


    
def load_img(f):
    f=open(f)
    lines=f.readlines()
    imgs, lab=[], []
    for i in range(len(lines)):
        fn, label = lines[i].split(' ')
        im1=cv2.imread(fn)
        im1=cv2.resize(im1, (256, 256))
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)#轉為灰階圖片
        
        blur = cv2.bilateralFilter(im1,9,25,25)
        blur2 = cv2.bilateralFilter(im1,12,25,25)
        blur3 = cv2.bilateralFilter(im1,15,25,25)
        blur4 = cv2.bilateralFilter(im1,18,25,25)
        blur5 = cv2.bilateralFilter(im1,21,25,25)
        
        im2 = custom_blur_demo(blur)
        im21 = custom_blur_demo(blur2)
        im22 = custom_blur_demo(blur3)
        im23 = custom_blur_demo(blur4)
        im24 = custom_blur_demo(blur5)
        
        im3 = im1-im2 #原始圖-模糊化(低通)=剩下高通
        im31 = im2 - im21 #模糊化-模糊化2
        im32 = im21 - im22 #模糊化2-模糊化3
        im33 = im22 - im23 #模糊化3-模糊化4
        im34 = im23 - im24 #模糊化4-模糊化5
        im35 = im22 - im24 #模糊化3-模糊化5
        imx = im35+im31
        
        im2+=(2*imx) #增強邊緣
        mg_mix = cv2.addWeighted(im2, 0.2, blur, 0.8, 0)#圖片融合
        
        img = np.array(mg_mix)
        mean = np.mean(img) #取平均
        img = img - mean
        img = img*0.8 + mean*0.9 #修正對比和亮度
        img = img/255.   #没有會白屏
        cv2.waitKey(0)
        '''===============================
        影像處理的技巧可以放這邊，來增強影像的品質
        
        ==============================='''
        
        vec = np.reshape(img, [-1])
        imgs.append(vec) 
        lab.append(int(label))
        
    imgs= np.asarray(imgs, np.float32)
    lab= np.asarray(lab, np.int32)
    return imgs, lab 


# ~ x, y = load_img('train.txt')
# ~ tx, ty = load_img('test.txt')
# ~ np.savez("finalall.npz", x=x,y=y,tx=tx,ty=ty)

# ~ np.save("endoftraing.npy", [x,y,tx,ty])
file = np.load('finalall.npz')
# ~ asd = np.load('asd.npz')

# ~ tttest = np.load("endoftraing.npy")
#======================================
#X就是資料，Y是Label，請設計不同分類器來得到最高效能
#必須要計算出分類的正確率
#======================================
def test(img):
    imgs, lab=[], []
    fn = img
    im1=cv2.imread(fn)
    im1=cv2.resize(im1, (256, 256))
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)#轉為灰階圖片
        
    blur = cv2.bilateralFilter(im1,9,25,25)
    blur2 = cv2.bilateralFilter(im1,12,25,25)
    blur3 = cv2.bilateralFilter(im1,15,25,25)
    blur4 = cv2.bilateralFilter(im1,18,25,25)
    blur5 = cv2.bilateralFilter(im1,21,25,25)
        
    im2 = custom_blur_demo(blur)
    im21 = custom_blur_demo(blur2)
    im22 = custom_blur_demo(blur3)
    im23 = custom_blur_demo(blur4)
    im24 = custom_blur_demo(blur5)
        
    im3 = im1-im2 #原始圖-模糊化(低通)=剩下高通
    im31 = im2 - im21 #模糊化-模糊化2
    im32 = im21 - im22 #模糊化2-模糊化3
    im33 = im22 - im23 #模糊化3-模糊化4
    im34 = im23 - im24 #模糊化4-模糊化5
    im35 = im22 - im24 #模糊化3-模糊化5
    imx = im35+im31
    
        
    im2+=(4*imx) #增強邊緣
    mg_mix = cv2.addWeighted(im2, 0.2, blur, 0.8, 0)#圖片融合
        
    img = np.array(mg_mix)
    mean = np.mean(img) #取平均
    img = img - mean
    img = img*0.8 + mean*0.9 #修正對比和亮度
    img = img/255.   #没有會白屏
    cv2.waitKey(0)
        
    vec = np.reshape(img, [-1])
    imgs.append(vec) 
        
    imgs= np.asarray(imgs, np.float32)
    return imgs

def originalImage(img):
    imgs, lab=[], []
    fn = img
    im1=cv2.imread(fn)
    im1=cv2.resize(im1, (256, 256))
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)#轉為灰階圖片

    cv2.waitKey(0)
        
    vec = np.reshape(im1, [-1])
    imgs.append(vec) 
        
    imgs= np.asarray(imgs, np.float32)
    return imgs


from sklearn.ensemble import RandomForestClassifier

x , y = file['x'] , file['y']
tx , ty =file['tx'] , file['ty']

model = RandomForestClassifier(n_estimators=1000,max_depth=35000,min_samples_split =3, min_samples_leaf =1)
model.fit(x, y)

# ~ py = model.predict(test('TIN/n01774750/images/n01774750_99.JPEG'))
py = model.predict(tx) #測試資料做預測
py2 = model.predict(originalImage('TIN/n01443537/images/n01443537_99.JPEG'))
# ~ qq = model.predict_proba(test('TIN/n01774750/images/n01774750_99.JPEG'))
qq = model.predict_proba(tx)
print(qq)
print(py)
print(py2)


