#!/usr/bin/env python
import rospy
import time

from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion

Imu_msg = None
globalData = None

def imu_callback(data):
    global Imu_msg, globalData
    Imu_msg = [data.orientation.x, data.orientation.y, data.orientation.z,
               data.orientation.w] 
    globalData = data

rospy.init_node("Imu_Print")
rospy.Subscriber("imu", Imu, imu_callback)

while not rospy.is_shutdown():
    if Imu_msg == None:
        continue
    
    (roll, pitch, yaw) = euler_from_quaternion(Imu_msg)
 
    print('Roll:%.4f, Pitch:%.4f, Yaw:%.4f' % (roll, pitch, yaw))
    print("Data : ", globalData)
    print()

    time.sleep(1.0)
