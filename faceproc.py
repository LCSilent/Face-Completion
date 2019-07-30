import sys
import os
import cv2
import dlib
from glob import glob
from faceproc_noise import add_noise
input_dir = 'img_align_celeba/'
output_dir = 'celebA_crop/' # 原图像裁剪后存放的路径
output_dir_intial = 'celebA/'# 原图像变成128*128大小后的图像
output_dir_noise = 'celebA_noise/'# 原图像加噪声之后的图像位置

if not os.path.exists(output_dir):  # 创建文件夹的操作
    os.makedirs(output_dir)

if not os.path.exists(output_dir_intial):
    os.makedirs(output_dir_intial)

if not os.path.exists(output_dir_noise):
    os.makedirs(output_dir_noise)

detector = dlib.get_frontal_face_detector() # 使用dlib来检测人脸

index = 0
imgs = glob(os.path.join(input_dir,'*.jpg')) # 遍历文件夹下所有的图片，返回的是图片的相对路径
filename = 'location.txt'
for imgpath in imgs:
    print(imgpath)
    img = cv2.imread(imgpath)
    img1 = cv2.resize(img,(128,128)) # 裁剪过的图片裁剪的大小为128*128
    gray_img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) # 将图片转换为灰度图，供dlib使用
    # 使用detector进行人脸的检测，dets为返回的结果
    dets = detector(gray_img,1)
    for i,d in enumerate(dets):
        x1 = d.top() if d.top()>0 else 0
        y1 = d.bottom() if d.bottom() >0 else 0
        x2 = d.left() if d.left()>0 else 0
        y2 = d.right() if d.right()>0 else 0


        face = img1[x1:y1,x2:y2]
        face = cv2.resize(face,(64,64))

        cv2.imshow('image',face)

        cv2.imwrite(output_dir+str(index)+'.jpg',face)
        cv2.imwrite(output_dir_intial+str(index)+'.jpg',img1)

        img_noise = add_noise(img1, x1, y1, x2, y2, index)
        cv2.imwrite(output_dir_noise+str(index)+'.jpg',img_noise)

        index+=1

    key = cv2.waitKey(20) & 0xff
    if key==27:
        sys.exit(0)