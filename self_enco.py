import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import random

PIC_SIZE_X = 64
PIC_SIZE_Y = 64
PIC_CHANNEL = 3
PIC_SIZE = PIC_SIZE_X * PIC_SIZE_Y * PIC_CHANNEL
PIC_DIR = "1_o_pro/"
TEST_DIR = "1_o_pro/"

#图像拆分成左右两部分，并resize图片为一维数组
def resize(img):
    img_left = img[:, 0:PIC_SIZE_X, :]
    img_right = img[:, PIC_SIZE_X:2*PIC_SIZE_X, :]

    #归一化
    img_left = img_left / 255
    img_right = img_right / 255

    return img_left, img_right

#
def encode(img):
    
                
    return img
    

#
def decode(img):
    
            
    return img
    
#随机读取N张图片的第channel个颜色通道，加噪声，制成数据集
#N大于图片总数时，读取所有图片
def dataset(pic_dir, N):
    file_name = os.listdir(pic_dir)
    file_num = len(file_name)
    in_pic_arr = np.zeros((N, PIC_SIZE_X, PIC_SIZE_Y, PIC_CHANNEL))
    out_pic_arr = np.zeros((N, PIC_SIZE_X, PIC_SIZE_Y, PIC_CHANNEL))
    
    pic_range = 0
    if (N < file_num):
        rnd_num = random.sample(range(0, file_num), N)
        pic_range = N
    else :
        rnd_num = range(0, file_num)
        pic_range = file_num
        
    print("pic_num = ", rnd_num)

    #随机抽取N张图片
    for i in range(pic_range):
        index = rnd_num[i]
        img = cv.imread(pic_dir + file_name[index])

        #resize图片并保存
        img_l, img_r = resize(img)
        noise = 0.1 * np.random.random(size=[PIC_SIZE_X, PIC_SIZE_Y, PIC_CHANNEL])
        in_pic_arr[i] = encode(img_l) + noise
        out_pic_arr[i] = encode(img_r)

    return in_pic_arr, out_pic_arr


#定义自编码器
#定义自编码器
#三通道卷积
def enco_layer(rgb, channel):
    res_out = tf.layers.conv2d(rgb, channel, (3,3), padding='same', activation = tf.nn.relu)
    down_sample = tf.layers.max_pooling2d(res_out, (2,2), (2,2), padding='same')

    return res_out, down_sample

#三通道反卷积
def deco_layer(res_out, down_sample, channel):
    conv_r = tf.layers.conv2d_transpose(down_sample, channel, (2,2), strides=(2,2), padding='same', activation=tf.nn.relu)
    conv_g = tf.layers.conv2d_transpose(down_sample, channel, (2,2), strides=(2,2), padding='same', activation=tf.nn.relu)
    conv_b = tf.layers.conv2d_transpose(down_sample, channel, (2,2), strides=(2,2), padding='same', activation=tf.nn.relu)

    conv_concat = tf.concat((conv_r, conv_g, conv_b, res_out), axis=3)

    up_sample = tf.layers.conv2d(conv_concat, channel, (3,3), padding='same', activation = tf.nn.relu)

    return up_sample

    
#网络结构
inputs_ = tf.placeholder(tf.float32, (None, PIC_SIZE_X, PIC_SIZE_Y, PIC_CHANNEL), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, PIC_SIZE_X, PIC_SIZE_Y, PIC_CHANNEL), name='targets')

### Encoder
res_out_1, down_sample = enco_layer(inputs_, 64)
res_out_2, down_sample = enco_layer(down_sample, 64)
res_out_3, down_sample = enco_layer(down_sample, 64)
res_out_4, down_sample = enco_layer(down_sample, 64)

### Decoder
up_sample = deco_layer(res_out_4, down_sample, 64)
up_sample = deco_layer(res_out_3, up_sample, 64)
up_sample = deco_layer(res_out_2, up_sample, 64)
up_sample = deco_layer(res_out_1, up_sample, 64)

decoded_def = tf.layers.conv2d(up_sample, 3, (3,3), padding='same', activation = tf.nn.tanh)

print(decoded_def.shape)

net_out = decoded_def

loss = tf.square(net_out - targets_)
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(2e-5).minimize(cost)

sess = tf.Session()

variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#打印网络参数
total_var_num = 0
for str_num in variables:
    var_shape = str_num.shape
    var_num = 1
    for i in var_shape:
        var_num *= i 
    total_var_num += var_num
print(str(total_var_num) + "\n\n")

epochs = 510000
batch_size = 200
CH_NUM = 2
err_trace = []
test_err = []
saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())

#载入上一次保存的模型
USE_BEFORE = True
try:
    if USE_BEFORE:
        model_file=tf.train.latest_checkpoint('ckpt/')
        saver.restore(sess, model_file)
except:
    print("保存模型不可用\n\n")


batch_x, batch_y = dataset(PIC_DIR, 4)


for e in range(epochs):
    batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: batch_x, targets_: batch_y})
    
    print("Epoch: {}/{}...".format(e+1, epochs),
              "Training loss: {}".format(np.mean(batch_cost)))
    
    err_trace.append(np.mean(batch_cost))

    if (e % 40 == 1):
        #重新生成数据集
        batch_x, batch_y = dataset(PIC_DIR, 4)
        
    if (e % 200 == 1):
        #保存模型
        saver.save(sess,'ckpt/enco.ckpt',global_step=e)

    if (e % 1000 == 1):
        y_, x, y = sess.run([net_out, inputs_, targets_], feed_dict={inputs_: batch_x, targets_: batch_y})
        
        plt.subplot(2, 3, 2)
        plt.imshow(y_[0])
        plt.subplot(2, 3, 1)
        plt.imshow(x[0])
        plt.subplot(2, 3, 3)
        plt.imshow(y[0])
        plt.subplot(2, 3, 5)
        plt.imshow(y_[1])
        plt.subplot(2, 3, 4)
        plt.imshow(x[1])
        plt.subplot(2, 3, 6)
        plt.imshow(y[1])
        plt.show()

        '''
        plt.subplot(2, 1, 1)
        plt.plot(range(len(err_trace)), err_trace)
        plt.subplot(2, 1, 2)
        plt.plot(range(len(test_err)), test_err)
        plt.show()
        '''


        
#测试网络
#batch_xs, batch_ys = dataset(PIC_DIR, 3)
#encode_out = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})

#plt.subplot(1, 2, 1)
#plt.imshow(de_resize(encode_out[1]))
#plt.subplot(1, 2, 2)
#plt.imshow(de_resize(encode_out[1]))
#plt.show()







