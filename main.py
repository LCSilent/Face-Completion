import tensorflow as tf
import cv2
from glob import glob
import os
from PIL import Image
import numpy as np
from G_network import inference_G

batch_size = 32
n_ecoph = 100
Image_size = 128
scaling = 300
X_noise = tf.placeholder(tf.float32,[None,Image_size,Image_size,3]) #输入的噪声图片
X = tf.placeholder(tf.float32,[None,Image_size,Image_size,3])  #原图


imgs = glob(os.path.join("celebA","*.jpg"))


imgslen = len(imgs)

imgs_noise = glob(os.path.join("celebA_noise","*.jpg"))

imgs_noise_len = len(imgs_noise)

generator_images = inference_G(X_noise,batch_size) #输入噪声图片，经过G生成的图片
'''
将生成的图片和原图片分别放入判别器，得出概率，通过判别器的损失函数来更新

'''
#g_loss = tf.sqrt(2*tf.nn.l2_loss(tf.abs(X-generator_images))) / batch_size
g_loss = tf.reduce_mean(tf.abs(X-generator_images))   # 使用l1距离来求生成图片和原来图片之间的距离

global_step = tf.Variable(0)
learning_rate =tf.train.exponential_decay(0.0001,global_step,imgslen//batch_size,0.98,staircase=True)
train_op_g = tf.train.AdamOptimizer(learning_rate).minimize(g_loss,global_step)



  # 优化判别器的损失函数
#train_op_g = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(g_loss)   # 优化生成器的损失函数

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9

noise = glob(os.path.join("celebA_noise_test","*.jpg"))
noise1 = noise[0:batch_size]
img_noise = np.array([cv2.imread(image) for image in noise1])
isTrain = True
def train():
    with tf.Session(config=config) as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        saver = tf.train.Saver()
        if os.path.exists(os.path.join("model", 'model.ckpt')) is True:
            print('加载模型中')
            saver.restore(sess, os.path.join("model", 'model.ckpt'))

        for i in range(n_ecoph):
            for j in range(imgslen // batch_size):
                batch_f = imgs[j * batch_size:(j + 1) * batch_size]
                batch_imgs = np.array([cv2.resize(cv2.imread(images), (128, 128)) for images in batch_f])
                batch_n = imgs_noise[j * batch_size:(j + 1) * batch_size]
                batch_imgs_noise = np.array([cv2.imread(image) for image in batch_n])
                loss_g, _ = sess.run([g_loss, train_op_g], feed_dict={X: batch_imgs, X_noise: batch_imgs_noise})

                # loss,_ = sess.run([g_loss,train_op_g],
                #  feed_dict={X_noise:batch_imgs_noise,
                #          X:batch_imgs})

                print("%d: [%d // %d]  g_loss%f " % (i, j, (imgslen // batch_size), loss_g))

                if j % 100 == 0:
                    # 每过100次保存一个模型
                    step = str(i) + str(j)
                    saver.save(sess, os.path.join("model", 'model.ckpt'),global_step=j)
                    img = sess.run(generator_images, feed_dict={X_noise: img_noise})
                    for k in range(batch_size):
                        image = img[k, :, :, :]
                        cv2.imwrite("save2/" + str(i) + '_' + str(j) + '_' + str(k) + ".jpg", image)


# 加载模型，对噪声图进行修复并且按原来的名字保存
def sample():
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        checkpoint_dir = 'model'
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver = tf.train.Saver()
        index = 0
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess,os.path.join(checkpoint_dir,ckpt_name))
        data = glob(os.path.join('celebA_noise_test','*.jpg'))
        print(len(data))
        print(data[0][17:])

        for i in range(len(data)//batch_size):
            batch_img = data[i*batch_size:(i+1)*batch_size]
            batch = np.array([cv2.imread(img) for img in batch_img])
            b = sess.run(generator_images,feed_dict={X_noise:batch})
            
            for k in range(batch_size):
                image = b[k,:,:,:]
                cv2.imwrite('save4/'+data[index][17:],image)
                print('save...'+data[index][17:])
                index+=1


sample()