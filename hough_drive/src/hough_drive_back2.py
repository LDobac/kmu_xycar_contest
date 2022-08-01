#!/usr/bin/env python
# -*- coding: utf-8 -*-

#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import numpy as np
import cv2, math
import rospy, rospkg, time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_msgs.msg import xycar_motor
from math import *
import signal
import sys
import os
import random


import torch
from torch import nn
from torch.nn import functional as F


device = 'cuda'


class ResBlock(nn.Module):
    def __init__(self, channels, hidden_channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_prev = x
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(x_prev + self.bn2(self.conv2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1)
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1)
        self.conv5 = nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1)
        self.res_block1 = ResBlock(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        xs = []
        xs.append(self.leaky_relu(self.conv1(x))) # (3, 112, 112)
        xs.append(self.leaky_relu(self.conv2(xs[-1]))) # (64, 56, 56)
        xs.append(self.leaky_relu(self.conv3(xs[-1]))) # (64, 28, 28)
        xs.append(self.leaky_relu(self.conv4(xs[-1]))) # (64, 14, 14)
        xs.append(self.leaky_relu(self.conv5(xs[-1]))) # (64, 7, 7)
        xs.append(self.dropout(self.res_block1(xs[-1]))) # (64, 7, 7)
        return xs


class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_channels=64, skip_channels=64):
        super(Decoder, self).__init__()

        self.res_block1 = ResBlock(hidden_channels, hidden_channels)
        self.conv1 = nn.ConvTranspose2d(hidden_channels + skip_channels, hidden_channels, 4, 2, 1)
        self.conv2 = nn.ConvTranspose2d(hidden_channels + skip_channels, hidden_channels, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(hidden_channels + skip_channels, hidden_channels, 4, 2, 1)
        self.conv4 = nn.ConvTranspose2d(hidden_channels + skip_channels, hidden_channels, 4, 2, 1)

        self.conv_out1 = nn.Conv2d(hidden_channels + skip_channels, out_channels, 1)
        self.conv_out2 = nn.Conv2d(hidden_channels + skip_channels, out_channels, 1)
        self.conv_out3 = nn.Conv2d(hidden_channels + skip_channels, out_channels, 1)
        self.conv_out4 = nn.Conv2d(hidden_channels + skip_channels, out_channels, 1)
        self.conv_out5 = nn.Conv2d(hidden_channels + skip_channels, out_channels, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, xs):
        x = xs[-1]
        
        x = torch.cat([xs[-2], self.res_block1(x)], dim=1) # (128, 7, 7)
        y = self.conv_out1(x) # (3, 7, 7)
        
        x = torch.cat([xs[-3], self.leaky_relu(self.conv1(x))], dim=1) # (128, 14, 14)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
        y = y + self.conv_out2(x) # (3, 14, 14)
        
        x = torch.cat([xs[-4], self.leaky_relu(self.conv2(x))], dim=1) # (128, 28, 28)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
        y = y + self.conv_out3(x) # (3, 28, 28)
        
        x = torch.cat([xs[-5], self.leaky_relu(self.conv3(x))], dim=1) # (128, 56, 56)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
        y = y + self.conv_out4(x) # (3, 56, 56)

        x = torch.cat([xs[-6], self.leaky_relu(self.conv4(x))], dim=1) # (128, 112, 112)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
        y = y + self.conv_out5(x) # (3, 112, 112)
        
        return y


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super(Model, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.encoder = Encoder(in_channels=in_channels, hidden_channels=hidden_channels)
        self.decoder = Decoder(out_channels=out_channels, hidden_channels=hidden_channels, skip_channels=hidden_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#=============================================
# 터미널에서 Ctrl-C 키입력으로 프로그램 실행을 끝낼 때
# 그 처리시간을 줄이기 위한 함수
#=============================================
def signal_handler(sig, frame):
    import time
    time.sleep(3)
    os.system('killall -9 python rosout')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0]) # 카메라 이미지를 담을 변수
bridge = CvBridge() # OpenCV 함수를 사용하기 위한 브릿지 
motor = None # 모터 토픽을 담을 변수
img_ready = False # 카메라 토픽이 도착했는지의 여부 표시 

#=============================================
# 프로그램에서 사용할 상수 선언부
#=============================================
CAM_FPS = 30    # 카메라 FPS - 초당 30장의 사진을 보냄
WIDTH, HEIGHT = 640, 480    # 카메라 이미지 가로x세로 크기
ROI_ROW = 250   # 차선을 찾을 ROI 영역의 시작 Row값 
ROI_HEIGHT = HEIGHT - ROI_ROW   # ROI 영역의 세로 크기  
L_ROW = ROI_HEIGHT - 120  # 차선의 위치를 찾기 위한 기준선(수평선)의 Row값

#=============================================
# 콜백함수 - 카메라 토픽을 처리하는 콜백함수
# 카메라 이미지 토픽이 도착하면 자동으로 호출되는 함수
# 토픽에서 이미지 정보를 꺼내 image 라는 변수에 옮겨 담음.
# 카메라 토픽의 도착을 표시하는 img_ready 값을 True로 바꿈.
#=============================================
def img_callback(data):
    global image, img_ready
    image = bridge.imgmsg_to_cv2(data, "bgr8")
    img_ready = True
    
#=============================================
# 모터 토픽을 발행하는 함수  
# 입력으로 받은 angle과 speed 값을 
# 모터 토픽에 옮겨 담은 후에 토픽을 발행함.
#=============================================
def drive(angle, speed):
    global motor
    motor_msg = xycar_motor()
    motor_msg.angle = angle
    motor_msg.speed = speed
    motor.publish(motor_msg)

def softmax(x, temperature=1, axis=-1):
    exp = np.exp(x / temperature)
    return exp / exp.sum(axis=axis, keepdims=True)

prev_xs = [[], []]
speed = 0
angle = 0

#=============================================
# 실질적인 메인 함수 
# 카메라 토픽을 받아 각종 영상처리와 알고리즘을 통해
# 차선의 위치를 파악한 후에 조향각을 결정하고,
# 최종적으로 모터 토픽을 발행하는 일을 수행함. 
#=============================================
def start():

    global image, img_ready, motor, prev_xs, speed, angle

    #=========================================
    # ROS 노드를 생성하고 초기화 함.
    # 카메라 토픽을 구독하고 모터 토픽을 발행할 것임을 선언
    #=========================================
    rospy.init_node('h_drive')
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    image_sub = rospy.Subscriber("/usb_cam/image_raw/",Image,img_callback)

    print ("----- Xycar self driving -----")

    # 첫번째 카메라 토픽이 도착할 때까지 기다림.
    while not image.size == (WIDTH * HEIGHT * 3):
        continue
 
    #=========================================
    # 메인 루프 
    # 카메라 토픽이 도착하는 주기에 맞춰 한번씩 루프를 돌면서 
    # "이미지처리 +차선위치찾기 +조향각결정 +모터토픽발행" 
    # 작업을 반복적으로 수행함.
    #=========================================
    while not rospy.is_shutdown():

        # 카메라 토픽이 도착할때까지 잠시 기다림
        while img_ready == False:
            continue
            
        img = image.copy()  # 이미지처리를 위한 카메라 원본이미지 저장
        display_img = img  # 디버깅을 위한 디스플레이용 이미지 저장
        img_ready = False  # 카메라 토픽이 도착하면 콜백함수 안에서 True로 바뀜
        
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.erode(img, kernel, iterations=3)
        img = cv2.GaussianBlur(img, (7, 7), 0.0)

        palette = np.array([[120, 130, 120], [150, 170, 160]])

        img = img.astype(np.float32) / 255
        palette = palette.astype(np.float32) / 255

        x = softmax(-np.square(img[..., None, :] - palette[None, None]).sum(-1), 0.05)

        x = x[..., 0]
        x = cv2.GaussianBlur(x, (7, 7), 0.0)
        x = x[300:]
        x = np.abs(cv2.Sobel(x, cv2.CV_64F,1,0,ksize=3)) * 2
        x = np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)
        x = cv2.GaussianBlur(x, (7, 7), 0.0)
        edges = cv2.Canny(x, 100, 200)

        # lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        
        vis = np.maximum(img[300:], np.tile(edges[..., None], (1, 1, 3)))

        # if lines is not None:
        #     for i in xrange(len(lines)):
        #         for rho, theta in lines[i]:
        #             a = np.cos(theta)
        #             b = np.sin(theta)
        #             x0 = a * rho
        #             y0 = b * rho
        #             x1 = int(x0 + 1000 * (-b))
        #             y1 = int(y0 + 1000 * (a))
        #             x2 = int(x0 - 1000 * (-b))
        #             y2 = int(y0 - 1000 * (a))
        #             cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        lines = cv2.HoughLinesP(edges, 1, math.pi/180, 50, 30, 10)
        if lines is not None:

            # seeds = [[0, img.shape[0] - 1], [img.shape[1] - 1, img.shape[0] - 1]]
            seeds = np.array([0, img.shape[1] - 1])
            xs = [[seeds[0]], [seeds[1]]]
            thetas = []
            for i in xrange(len(lines)):
                for x1, y1, x2, y2 in lines[i]:
                    x = float(x1 + x2) * 0.5
                    argmin = np.square(seeds - x).argmin()
                    color = (0, 0, 255) if argmin == 0 else (255, 0, 0)

                    if np.abs(y2 - y1) < 2:
                        continue
                    thetas.append(np.arctan((x2 - x1) / (y2 - y1)))
                    vis = cv2.line(vis, (x1, y1), (x2, y2), color, 2)
                    xs[argmin].append(x)
            mid = (np.median(xs[0]) + np.median(xs[1])) * 0.5
            cv2.line(vis, (int(mid), 0), (int(mid), 1000), (0, 255, 0), 3)
            
            speed = 4
            # angle = -15 if ((640.0 - 1.0) * 0.5 - mid) * 0.1 < 0 else 15
            # angle = -20 if len(xs[0]) < len(xs[1]) else 20
            # angle = -30 if np.mean(thetas) > 0 else 30
            # angle += -15 if ((640.0 - 1.0) * 0.5 - mid) * 0.1 <0 else 15
            print(mid)
            print(np.mean(thetas))
            angle = np.clip(np.mean(thetas) * -30 + ((640.0 - 1.0) * 0.5 - mid) * -0.5, -50, 50)
        

        cv2.imshow("img", img)
        cv2.imshow("vis", vis)
        cv2.waitKey(1)
        
        drive(angle, speed)
        

#=============================================
# 메인 함수
# 가장 먼저 호출되는 함수로 여기서 start() 함수를 호출함.
# start() 함수가 실질적인 메인 함수임. 
#=============================================
if __name__ == '__main__':
    start()

