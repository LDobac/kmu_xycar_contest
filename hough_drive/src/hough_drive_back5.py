#!/usr/bin/env python
# -*- coding: utf-8 -*-

#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
from imp import is_builtin
import numpy as np
import cv2, math
import rospy, rospkg, time
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from sensor_msgs.msg import LaserScan
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


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.bn(self.conv(x))
        x = F.relu(x, inplace=True)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, use_relu=True):
        super(UpConv, self).__init__()
        
        self.use_batchnorm = use_batchnorm
        self.use_relu = use_relu

        self.conv = nn.ConvTranspose2d(in_channels + skip_channels, out_channels, 4, 2, 1, bias=not use_batchnorm)
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, c=None):
        if c is not None:
            x = torch.cat([x, c], dim=1)
        
        x = self.conv(x)
        
        if self.use_batchnorm:
            x = self.bn(x)
        if self.use_relu:
            x = F.relu(x, inplace=True)
            
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, hidden_channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x_prev = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(x_prev + self.bn2(self.conv2(x)), inplace=True)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=16):
        super(Encoder, self).__init__()
        
        self.conv1 = DownConv(in_channels, hidden_channels) # 16x28x28
        self.conv2 = DownConv(hidden_channels, hidden_channels * 2) # 32x14x14
        self.conv3 = DownConv(hidden_channels * 2, hidden_channels * 4) # 64x7x7
        self.resblk1 = ResBlock(hidden_channels * 4, hidden_channels * 4)
        self.resblk2 = ResBlock(hidden_channels * 4, hidden_channels * 4)

    def forward(self, x):
        xs = []
        xs.append(self.conv1(x))
        xs.append(self.conv2(xs[-1]))
        xs.append(self.resblk2(self.resblk1(self.conv3(xs[-1]))))
        return xs


class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_channels=16):
        super(Decoder, self).__init__()
        
        self.conv1 = UpConv(hidden_channels * 4, 0, hidden_channels * 2) # 32x14x14
        self.conv2 = UpConv(hidden_channels * 2, hidden_channels * 2, hidden_channels) # 16x28x28
        self.conv3 = UpConv(hidden_channels, hidden_channels, 3, use_batchnorm=False, use_relu=False) # 3x56x56

    def forward(self, xs):
        x = self.conv1(xs[-1])
        x = self.conv2(x, xs[-2])
        x = self.conv3(x, xs[-3])
        return x


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=16):
        super(Model, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.encoder = Encoder(in_channels=in_channels, hidden_channels=hidden_channels)
        self.decoder = Decoder(out_channels=out_channels, hidden_channels=hidden_channels)

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


lidar_points = None

def lidar_callback(data):
    global lidar_points
    lidar_points = data.ranges


class LidarManager:
    def __init__(self):
        self.window_size = 20
        self.distance_threshold = 0.1
        self.dot_threshold = 0.8
        self.cluster_threshold = 5

    def get_points(self, data, lidar_angle_offset=0):
        radians = np.deg2rad(np.linspace(180, 0, 180) + lidar_angle_offset)
        distances = np.array(data[11:11 + 180], dtype=np.float32)
        points = np.stack([np.cos(radians), np.sin(radians)], axis=1) * distances.reshape(-1, 1)
        return points

    def find_walls(self, points):
        walls = []
        grads = []
        del_points = np.zeros((len(points), ), dtype=np.bool)
        for i in range(len(points)):
            if del_points[i]:
                continue
            indices = [[i]]
            j = i
            grad = None
            while True:
                other_points = points[j + 1:j + self.window_size + 1]
                dists = np.linalg.norm(other_points - points[j:j + 1], axis=1)
                new_indices = np.nonzero(dists < self.distance_threshold)[0]
                if len(new_indices) == 0:
                    break
                other_points = other_points[new_indices]
                curr_grad = np.mean(np.concatenate([[other_points[0] - points[j]], other_points[1:] - other_points[:-1]], axis=0), axis=0)
                curr_grad = curr_grad / np.linalg.norm(curr_grad)
                if grad is None:
                    grad = curr_grad
                if np.dot(curr_grad, grad) < self.dot_threshold:
                    break
                indices.append(new_indices + (j + 1))
                j = new_indices[-1] + (j + 1)
            if len(indices) >= self.cluster_threshold:
                indices = np.concatenate(indices)
                del_points[indices] = True
                walls.append(indices)
                grads.append(grad)

        return walls, grads

    def visualize_walls(self, walls, colors, points, img_size=(200, 100), points_range=(-3, 0, 3, 3)):
        img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

        for point in points:
            if not (points_range[0] < point[0] < points_range[2] and points_range[1] < point[1] < points_range[3]):
                continue
            x = int(round(((point[0] - points_range[0]) / (points_range[2] - points_range[0])) * (img.shape[1] - 1)))
            y = int(round((1 - (point[1] - points_range[1]) / (points_range[3] - points_range[1])) * (img.shape[0] - 1)))
            img = cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

        for wall, color in zip(walls, colors):
            wall = points[wall]
            for point in wall:
                x = int(round(((point[0] - points_range[0]) / (points_range[2] - points_range[0])) * (img.shape[1] - 1)))
                y = int(round((1 - (point[1] - points_range[1]) / (points_range[3] - points_range[1])) * (img.shape[0] - 1)))
                img = cv2.circle(img, (x, y), 2, tuple(np.round(color * 255).astype(int)[::-1]), -1)
            
        return img

class States:
    def __init__(self):
        self.inside_tunnel = False
        self.tunnel_left = False

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
    rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)

    dst_size = (81, 81)
    padding = (22, -40, 22, 6)

    src = ((np.array([
        [286, 277],
        [0, 406],
        [376, 277],
        [639, 400]
    ], dtype=np.float32) - [80, 0]) * (56.0 / 480.0)).astype(np.float32)
    dst = (np.array([
        [padding[0], padding[1]],
        [padding[0], dst_size[1] - padding[3] - 1],
        [dst_size[0] - padding[2] - 1, padding[1]],
        [dst_size[0] - padding[2] - 1, dst_size[1] - padding[3] - 1]
    ], dtype=np.float32) * (81.0 / 80.0)).astype(np.float32)
    T = cv2.getPerspectiveTransform(src, dst)

    lidar_mng = LidarManager()
    states = States()

    print('----- Initializing Deep Network -----')

    model = Model(3, 3, hidden_channels=16).to(device)
    model.load_state_dict(torch.load('/home/nvidia/legacy2.ckpt')['model_state_dicts']['model'])
    model.eval()

    print ("----- Xycar self driving -----")

    from time import time

    prev_time = -1
    video_index = 0
    frame_index = 0

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
        if imu is None:
            continue
        if lidar_points is None:
            continue
            
        try:
            with open('/home/nvidia/fine_tuning.txt') as txt_file:
                lidar_angle_offset, tunnel_front_threshold, tunnel_left_amount, y_range_start, y_range_end, lr_offset, left_coeff, right_coeff, P_coeff, D_coeff, I_coeff, P_seed, C_speed = np.array(txt_file.read().split()).astype(float)
        except:
            lidar_angle_offset = tunnel_front_threshold = tunnel_left_amount = 0
            y_range_start = 15
            y_range_end = 0
            lr_offset = left_coeff = right_coeff = P_coeff = D_coeff = I_coeff = P_seed = C_speed = 0
        
        # img = cv2.resize(image, (int(image.shape[1] * (300.0 / image.shape[0])), 300)) # image.copy()  # 이미지처리를 위한 카메라 원본이미지 저장
        img = image.copy()
        img_56 = cv2.resize(img, (56, 56))
        
        crop_size = 56
        y_offset = (img_56.shape[0] - crop_size) // 2
        x_offset = (img_56.shape[1] - crop_size) // 2
        img_56 = img_56[y_offset:y_offset + crop_size, x_offset:x_offset + crop_size].astype(np.float32)
        img_56 /= 256

        with torch.no_grad():
            x = torch.from_numpy(img_56[:, :, ::-1].copy()).to(device) * 2 - 1
            x = x.permute(2, 0, 1).unsqueeze(0)
            x = model(x)
            x = F.softmax(x, dim=1)
            x = x.squeeze(0).permute(1, 2, 0)
            x = x.cpu().numpy() #[:, :, ::-1]
            
        x_warp = cv2.warpPerspective(x, T, dst_size)
            
        y_range = (56 - int(y_range_start), 56 - int(y_range_end))
            
        left = np.cumsum(x_warp[y_range[0]:y_range[1], ::-1, 0], axis=1)[:, ::-1]
        right = np.cumsum(x_warp[y_range[0]:y_range[1], :, 1], axis=1)
        fallback = np.zeros(left.shape, dtype=np.float32)
        fallback[:, :40] = 0.01
        fallback[:, 40:] = -0.01
        mix = np.sign(left - right + fallback)
        error = mix.sum() / (y_range[1] - y_range[0])
        # print(error, time() - prev_time)

        #vis = cv2.resize(img_112 * 0.5 + x * 0.5, (224, 224))
        
        # if imu is not None:
        #     cv2.imwrite('/home/nvidia/output/cam_{:06}.png'.format(frame_index), img)
        #     with open('/home/nvidia/output/imu_{:06}.txt'.format(frame_index), 'w') as txt_file:
        #         txt_file.write(str(time()) + '\n')
        #         for value in imu:
        #             txt_file.write(str(value) + ' ')
        #         txt_file.write('\n')
        #         for value in lidar_points:
        #             txt_file.write(str(value) + ' ')
        #     frame_index += 1
        #     pass
        
        #lidar = np.zeros((360, ), dtype=np.float32)
        #lidar[270:] = lidar_points[:90]
        #lidar[:270] = lidar_points[90:][::-1]
        #lidar = np.array(lidar_points).astype(np.float32)[:180]
        #print(lidar.min(), lidar.max())
        #print(lidar)

        # lidar_vis = np.zeros((100, 200, 3), dtype=np.uint8)
        # x_range = (-2.0, 2.0)
        # y_range = (0.0, 2.0)

        # def to_pixels_x(px):
        #     return int(round(((px - x_range[0]) / (x_range[1] - x_range[0])) * (lidar_vis.shape[1] - 1)))
        # def to_pixels_y(py):
        #     return int(round((1 - (py - y_range[0]) / (y_range[1] - y_range[0])) * (lidar_vis.shape[0] - 1)))

        # for px in np.arange(x_range[0], x_range[1], 0.2):
        #     px = to_pixels_x(px)
        #     lidar_vis = cv2.line(lidar_vis, (px, 0), (px, lidar_vis.shape[1] - 1), (255, 255, 255), 1)
        # for py in np.arange(y_range[0], y_range[1], 0.2):
        #     py = to_pixels_y(py)
        #     lidar_vis = cv2.line(lidar_vis, (0, py), (lidar_vis.shape[1] - 1, py), (255, 255, 255), 1)

        # tunnel_wall_x_range = (-0.55, 0.55)
        # tunnel_front_x_range = (-0.2, 0.2)
        # tunnel_wall_threshold = 40
        # tunnel_front_threshold = 0.6

        # px1 = to_pixels_x(tunnel_wall_x_range[0])
        # px2 = to_pixels_x(tunnel_wall_x_range[1])
        # py = to_pixels_y(tunnel_front_threshold)
        # lidar_vis = cv2.line(lidar_vis, (px1, py), (px1, lidar_vis.shape[1] - 1), (0, 255, 0), 1)
        # lidar_vis = cv2.line(lidar_vis, (px2, py), (px2, lidar_vis.shape[1] - 1), (0, 255, 0), 1)
        # lidar_vis = cv2.line(lidar_vis, (px1, py), (px2, py), (0, 255, 0), 1)

        # pts = np.deg2rad(np.linspace(180, 0, 180))
        # pts = np.stack([np.cos(pts), np.sin(pts)], axis=1) * lidar.reshape(-1, 1)

        # left_pts = (pts[:, 0] < tunnel_front_x_range[0]).sum() - (pts[:, 0] < tunnel_wall_x_range[0]).sum()
        # right_pts = (pts[:, 0] > tunnel_front_x_range[1]).sum() - (pts[:, 0] > tunnel_wall_x_range[1]).sum()
        
        # if left_pts > tunnel_wall_threshold and right_pts > tunnel_wall_threshold:
        #     front_pts = (pts[:, 0] > tunnel_front_x_range[1]).sum() - (pts[:, 0] > tunnel_front_x_range[0]).sum()
        #     print('inside a tunnel', front_pts)

        #     if front_pts > tunnel_wall_threshold:
        #         print('turn left')

        # for px, py in pts:
        #     if not (x_range[0] < px < x_range[1] and y_range[0] < py < y_range[1]):
        #         continue

        #     px = to_pixels_x(px)
        #     py = to_pixels_y(py)

        #     lidar_vis = cv2.circle(lidar_vis, (int(px), int(py)), 1, (0, 0, 255), -1)

        angle = error * P_coeff + lr_offset
        if angle < 0:
            angle *= left_coeff
        else:
            angle *= right_coeff
            
        speed = abs(error) * P_seed + C_speed
        
        # states.inside_tunnel = False

        # if np.sum(np.array(lidar_points) < 1) > 30:
        #     points = lidar_mng.get_points(lidar_points, lidar_angle_offset)
        #     walls, grads = lidar_mng.find_walls(points)

        #     walls_tunnel = []
        #     grads_tunnel = []
        #     lengths_tunnel = []
        #     for wall, grad in zip(walls, grads):
        #         wall_points = points[wall]
        #         wall_length = np.linalg.norm(wall_points[-1] - wall_points[0])
        #         #if wall_length < 0.1:
        #         #    continue
        #         if np.abs(wall_points[:, 0]).min() > 0.55:
        #             continue
        #         walls_tunnel.append(wall)
        #         grads_tunnel.append(grad)
        #         lengths_tunnel.append(wall_length)

        #     grads_tunnel = np.array(grads_tunnel)
        #     lengths_tunnel = np.array(lengths_tunnel)

        #     tunnel_front = -1
        #     tunnel_left = -1
        #     tunnel_right = -1

        #     for i in np.argsort(-lengths_tunnel):
        #         if tunnel_front < 0 and grads_tunnel[i][0] > 0.8:
        #             tunnel_front = i
        #             continue
        #         if tunnel_left < 0 and grads_tunnel[i][1] > 0.8:
        #             tunnel_left = i
        #             continue
        #         if tunnel_right < 0 and grads_tunnel[i][1] < -0.8:
        #             tunnel_right = i
        #             continue

        #     states.inside_tunnel = len(walls_tunnel) > 0
        #     if states.inside_tunnel:
        #         if tunnel_front >= 0 and np.linalg.norm(points[walls_tunnel[tunnel_front]], axis=1).min() < tunnel_front_threshold:
        #             states.tunnel_left = True
                
        #         walls_vis = lidar_mng.visualize_walls(walls, np.concatenate([np.array(grads) * 0.5 + 0.5, np.zeros((len(grads), 1))], axis=1), points)

        #         vis_targets = np.array([tunnel_front, tunnel_left, tunnel_right])
        #         vis_colors = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        #         vis_colors = vis_colors[vis_targets >= 0]
        #         vis_targets = vis_targets[vis_targets >= 0]
        #         walls_tunnel_target = []
        #         for i in vis_targets:
        #             walls_tunnel_target.append(walls_tunnel[i])
        #         walls_tunnel_vis = lidar_mng.visualize_walls(walls_tunnel_target, vis_colors, points)

        #         cv2.imshow("walls", walls_vis)
        #         cv2.imshow("walls_tunnel", walls_tunnel_vis)

        # print(states.inside_tunnel, states.tunnel_left)

        # if not states.inside_tunnel:
        #     states.tunnel_left = False

        # angle = 0
        # if states.inside_tunnel and states.tunnel_left:
        #     angle = tunnel_left_amount
        # speed = 5

        drive(angle, speed)
        
        #cv2.imshow("img", img)
        #cv2.imshow("x", x[..., ::-1])
        #cv2.imshow("x_warp", x_warp[..., ::-1])
        #cv2.imshow("mix", mix * 0.5 + 0.5)
        # cv2.imshow("lidar", 1 - np.clip(np.tile((lidar / 2).reshape(1, -1), (30, 1)), 0, 1))
        #cv2.imshow("lidar_vis", lidar_vis)
        cv2.waitKey(1)
        
        #print(time() - prev_time)
        
        prev_time = time()
        
    #out.release()
        

#=============================================
# 메인 함수
# 가장 먼저 호출되는 함수로 여기서 start() 함수를 호출함.
# start() 함수가 실질적인 메인 함수임. 
#=============================================
if __name__ == '__main__':
    start()

