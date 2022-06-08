# coding=utf-8
import cv2
import os
from cv2 import VideoCapture
from cv2 import imwrite
def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    imwrite(address, image)


if __name__ == '__main__':

    video_path = "./test/test-heng.mp4"  # 视频路径
    out_path = "./output/img_"  # 保存图片路径+名字

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
        time_interval = 1

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

file_dir = './output/'
list = []
for root, dirs, files in os.walk(file_dir):
    for file in files:
        list.append(file)  # 获取目录下文件名列表

# VideoWriter是cv2库提供的视频保存方法，将合成的视频保存到该路径中
# 'MJPG'意思是支持jpg格式图片
# fps = 5代表视频的帧频为5，如果图片不多，帧频最好设置的小一点
# (1280,720)是生成的视频像素1280*720，一般要与所使用的图片像素大小一致，否则生成的视频无法播放
# 定义保存视频目录名称和压缩格式，像素为1280*720
video = cv2.VideoWriter('C:/Users/IAN/Desktop/test.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (1920, 1080))

for i in range(1, len(list)):
    print(list[i-1])
    # 读取图片
    img = cv2.imread('./output/' + list[i - 1])
    # resize方法是cv2库提供的更改像素大小的方法
    # 将图片转换为1280*720像素大小
    # 写入视频
    video.write(img)

# 释放资源
video.release()

