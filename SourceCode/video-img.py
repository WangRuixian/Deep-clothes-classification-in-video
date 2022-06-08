# -*- coding: utf-8 -*-

import os

import numpy as np
import time
import tensorflow as tf
import cv2
from cv2 import VideoCapture
from cv2 import imwrite

from utils_my import Read_Img_2_Tensor, Load_DeepFashion2_Yolov3, Draw_Bounding_Box


def Detect_Clothes(img, model_yolov3, eager_execution=True):
    """Detect clothes in an image using Yolo-v3 model trained on DeepFashion2 dataset"""
    img = tf.image.resize(img, (416, 416))

    t1 = time.time()
    if eager_execution == True:
        boxes, scores, classes, nums = model_yolov3(img)
        # change eager tensor to numpy array
        boxes, scores, classes, nums = boxes.numpy(), scores.numpy(), classes.numpy(), nums.numpy()
    else:
        boxes, scores, classes, nums = model_yolov3.predict(img)
    t2 = time.time()
    print('Yolo-v3 feed forward: {:.2f} sec'.format(t2 - t1))

    class_names = ['short_sleeve_top', 'long_sleeve_top', 'short_sleeve_outwear', 'long_sleeve_outwear',
                   'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeve_dress',
                   'long_sleeve_dress', 'vest_dress', 'sling_dress']

    # Parse tensor
    list_obj = []
    for i in range(nums[0]):
        obj = {'label': class_names[int(classes[0][i])], 'confidence': scores[0][i]}
        obj['x1'] = boxes[0][i][0]
        obj['y1'] = boxes[0][i][1]
        obj['x2'] = boxes[0][i][2]
        obj['y2'] = boxes[0][i][3]
        list_obj.append(obj)

    return list_obj


def Detect_Clothes_and_Crop(img_tensor, model, threshold=0.5):
    list_obj = Detect_Clothes(img_tensor, model)

    img = np.squeeze(img_tensor.numpy())
    img_width = img.shape[1]
    img_height = img.shape[0]

    # crop out one cloth
    for obj in list_obj:
        if obj['label'] == 'short_sleeve_top' and obj['confidence'] > threshold:
            img_crop = img[int(obj['y1'] * img_height):int(obj['y2'] * img_height),
                       int(obj['x1'] * img_width):int(obj['x2'] * img_width), :]

    return img_crop


def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    imwrite(address, image)


if __name__ == '__main__':

    video_path = "./test/test-heng.mp4"  # 视频路径
    out_path = "./output/transit/img_"  # 保存图片路径+名字
    is_all_frame = False  # 是否取所有的帧
    sta_frame = 1  # 开始帧
    end_frame = 120  # 结束帧
    ######
    time_interval = 1  # 时间间隔
    # 读取视频文件
    videoCapture = VideoCapture(video_path)
    # 读帧
    success, frame = videoCapture.read()
    print(success)
    i = 0
    j = 0
    if is_all_frame:
        time_interval = 2
    while success:
        i = i + 1
        if (i % time_interval == 0):
            if is_all_frame == False:
                if i >= sta_frame and i <= end_frame:
                    j = j + 1
                    print('save frame:', i+1000)
                    save_image(frame, out_path, j+1000)
                elif i > end_frame:
                    break
            else:
                j = j + 1
                print('save frame:', i+1000)
                save_image(frame, out_path, j+1000)

        success, frame = videoCapture.read()

    fileList = os.listdir(r'./output/transit')
    for name in fileList:
        img = Read_Img_2_Tensor('./output/transit' + "/" + name)
        print(img)
        model = Load_DeepFashion2_Yolov3()
        list_obj = Detect_Clothes(img, model)
        img_with_boxes = Draw_Bounding_Box(img, list_obj)

        #cv2.imshow("Clothes detection", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
        #cv2.waitKey(1)
        # cv2.destroyAllWindows()
        cv2.imwrite("./output/final/"+name, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)*255)

    file_dir = './output/final/'
    list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            list.append(file)  # 获取目录下文件名列表

    # VideoWriter是cv2库提供的视频保存方法，将合成的视频保存到该路径中
    # 'MJPG'意思是支持jpg格式图片
    # fps = 5代表视频的帧频为5，如果图片不多，帧频最好设置的小一点
    # (1280,720)是生成的视频像素1280*720，一般要与所使用的图片像素大小一致，否则生成的视频无法播放
    # 定义保存视频目录名称和压缩格式，像素为1280*720
    video = cv2.VideoWriter('./output/test.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (1920, 1080))

    for i in range(1, len(list)):
        print(list[i - 1])
        # 读取图片
        img = cv2.imread('./output/final/' + list[i - 1])
        # resize方法是cv2库提供的更改像素大小的方法
        # 将图片转换为1280*720像素大小
        # 写入视频
        video.write(img)

    # 释放资源
    video.release()