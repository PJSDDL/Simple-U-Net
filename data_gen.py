import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

OUT_SIZE = 64
#PIC_DIR = "TEST"
PIC_DIR = "1_o"  
OUT_DIR = PIC_DIR + "_pro"

try:
    os.makedirs(OUT_DIR)
except :
    print ("路径已存在")

for i in range(100) :
    img_c = img_n = np.zeros((64, 64, 3))

    img_c = cv.circle(img_c, (32, int(i/5) + 32), int(i/10) + 10, (255, 255, 255), 5)

    #给图片添加高斯噪声
    gauss = np.random.normal(0, 100, (64, 64, 3))
    noisy_img = img_c + gauss
    img_n = np.clip(noisy_img,a_min=0,a_max=255)

    #拼接
    img_train = np.concatenate([img_n, img_c], axis=1)

    '''
    plt.imshow(img_train)
    plt.show()
    '''

    cv.imwrite(OUT_DIR + "/" + PIC_DIR + str(i) + ".png", img_train)

        
