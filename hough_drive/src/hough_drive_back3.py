#!/usr/bin/env python
# -*- coding: utf-8 -*-

#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import numpy as np
import cv2, math
import rospy, rospkg, time
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
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


imu = None

def imu_callback(data):
    global imu
    imu_data = []
    imu_data.append(data.linear_acceleration.x)
    imu_data.append(data.linear_acceleration.y)
    imu_data.append(data.linear_acceleration.z)
    imu_data.append(data.orientation.w)
    imu_data.append(data.orientation.x)
    imu_data.append(data.orientation.y)
    imu_data.append(data.orientation.z)
    imu_data.append(data.angular_velocity.x)
    imu_data.append(data.angular_velocity.y)
    imu_data.append(data.angular_velocity.z)
    imu = imu_data

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
    imu_sub = rospy.Subscriber("imu", Imu, imu_callback)

    width = 128
    height = 256
    src = np.array([[-96, 223], [319, 223], [70, 140], [153, 140]], dtype=np.float32) / 223 * 111
    dst = np.array([[0, height - 1], [width - 1, height - 1], [0, 0], [width - 1, 0]], dtype=np.float32)
    #width, height = 90, 128
    #src = np.array([[11.30239486694336, 43.55845642089844], [-153.95802307128906, 111.0], [106.11070251464844, 36.76250457763672], [403.8610534667969, 111.0]]).astype('float32')
    #dst = np.array([[10.0, 10.0], [10.0, 127.0], [79.0, 10.0], [79.0, 127.0]]).astype('float32')
    T = cv2.getPerspectiveTransform(src, dst)

    mask = np.zeros((112, 112), dtype=np.float32)
    mask[10:-10, 10:-10] = 1
    mask = cv2.warpPerspective(mask, T, (width, height))

    print('----- Initializing Deep Network -----')

    model = Model(3, 3).to(device)
    model.load_state_dict(torch.load('/home/nvidia/checkpoint.ckpt')['model_state_dicts']['model'])
    model.eval()

    print ("----- Xycar self driving -----")

    from time import time

    prev_time = -1
    video_index = 0
    frame_index = 0
    
    errors = []
    error_times = []
    error_window_size = 30

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
        #if time() - prev_time > 100:
        #    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #    out = cv2.VideoWriter('/home/nvidia/output_{}.mp4'.format(video_index), fourcc, 30.0, (640, 480))
        #    video_index += 1
        #    prev_time = time()

        # 카메라 토픽이 도착할때까지 잠시 기다림
        while img_ready == False:
            continue
            
        # img = cv2.resize(image, (int(image.shape[1] * (300.0 / image.shape[0])), 300)) # image.copy()  # 이미지처리를 위한 카메라 원본이미지 저장
        img = image.copy()
        
        crop_size = 300
        y_offset = img.shape[0] - crop_size #(img.shape[0] - crop_size) // 2
        x_offset = (img.shape[1] - crop_size) // 2 + 30
        img_112 = img[y_offset:y_offset + crop_size, x_offset:x_offset + crop_size].astype(np.float32) / 255
        #img_112 = img[-crop_size - 50:-50, x_offset:x_offset + crop_size].astype(np.float32) / 255
        img_112 = cv2.resize(img_112, (112, 112))
	
        with torch.no_grad():
            x = torch.from_numpy(img_112[:, :, ::-1].copy()).to(device) * 2 - 1
            x = x.permute(2, 0, 1).unsqueeze(0)
            x = model(x)
            x = F.softmax(x, dim=1)
            x = x.squeeze(0).permute(1, 2, 0)
            x = x.cpu().numpy()[:, :, ::-1]
            # x[:60] = 0

        x_warp = cv2.warpPerspective(x, T, (width, height))
        x_edges = cv2.Laplacian(x_warp[:, :, 1] - x_warp[:, :, 2], cv2.CV_32F) * mask
        x_edges = np.round((x_edges - x_edges.min()) / (x_edges.max() - x_edges.min()) * 255).astype(np.uint8)
        x_edges = cv2.Canny(x_edges, 50, 150)

        lines = cv2.HoughLinesP(x_edges, 1, np.pi / 180, 30)
        vis_edges = x_warp.copy()
        angles = [0]
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                vis_edges = cv2.line(vis_edges, (x1, y1), (x2, y2), (1, 1, 1), 5)
                if abs(y2 - y1) < 1:
                    continue
                cur_angle = np.rad2deg(np.arctan(float(x2 - x1) / float(y2 - y1)))
                if cur_angle < 5:
                    continue
                angles.append(cur_angle)

        vis = cv2.resize(img_112 * 0.5 + x * 0.5, (224, 224))
        # out.write(np.round(vis * 255).astype(np.uint8))
        # out.write(img)

        if imu is not None:
            cv2.imwrite('/home/nvidia/output/cam_{:06}.png'.format(frame_index), img)
            with open('/home/nvidia/output/imu_{:06}.txt'.format(frame_index), 'w') as txt_file:
                txt_file.write(str(time()) + '\n')
                for value in imu:
                    txt_file.write(str(value) + ' ')
            #frame_index += 1

        left_raw = x[:, :, 2].sum()
        right_raw = x[:, :, 1].sum()
        
        total = left_raw + right_raw
        left = left_raw / total
        right = right_raw / total

        #print(left, right)
        
        #cv2.imshow("img", img)
        #cv2.imshow("vis", vis)
        #cv2.imshow("vis_cutoff", cv2.resize(img_112 * 0.5 + (x > 0.5) * 0.5, (224, 224)))
        #cv2.imshow("x_warp", x_warp)
        #cv2.imshow("x_edges", x_edges)
        #cv2.imshow('vis_edges', vis_edges)
        #cv2.imshow("left", x[:, :, 2])
        #cv2.imshow("right", x[:, :, 1])
        #cv2.waitKey(1)
        
        try:
            with open('/home/nvidia/fine_tuning.txt') as txt_file:
                left_coeff, right_ceoff, P_coeff, D_coeff, I_coeff, P_seed, C_speed = np.array(txt_file.read().split()).astype(float)
        except:
            left_coeff = right_ceoff = P_coeff = D_coeff = I_coeff = P_seed = C_speed = 0
        error = (np.median(angles) * -1 + (left * 50 + right * -50) * 0.3) #* 0.5
        
        errors.append(error)
        error_times.append(time())
        if len(errors) > error_window_size:
            del errors[0]
            del error_times[0]
            
        if len(errors) > 1:
            errors_arr = np.array(errors)
            times_arr = np.array(error_times)
            D = np.mean((errors_arr[1:] - errors_arr[:-1]) / (times_arr[1:] - times_arr[:-1]))
            I = np.sum(errors_arr[1:] * (times_arr[1:] - times_arr[:-1]))
        else:
            D = I = 0.0
            
        angle = P_coeff * error + D_coeff * D + I_coeff * I
        if C_speed < 0.1:
            speed = 0.0
        else:
            speed = C_speed + P_seed * abs(angle)
        
        if angle < 0:
            angle *= left_coeff # 0.6
        else:
            angle *= right_ceoff # 1.2
        #angle *= 1.2
        #speed = 10
        #print(len(angles), np.median(angles))
        
        #angle = speed = 0
        drive(angle, speed)

    out.release()
        

#=============================================
# 메인 함수
# 가장 먼저 호출되는 함수로 여기서 start() 함수를 호출함.
# start() 함수가 실질적인 메인 함수임. 
#=============================================
if __name__ == '__main__':
    start()

