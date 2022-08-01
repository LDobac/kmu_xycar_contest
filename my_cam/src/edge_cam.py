#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import rospy
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()
cv_image = np.empty(shape=[0])

def img_callback(data):
    global cv_image
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

rospy.init_node('cam_tune', anonymous=True)
rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)


def softmax(x, temperature=1, axis=-1):
    exp = np.exp(x / temperature)
    return exp / exp.sum(axis=axis, keepdims=True)

while not rospy.is_shutdown():

    if cv_image.size != (640*480*3):
        continue


    img = cv_image
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
    vis = np.tile(edges[..., None], (1, 1, 3))
    #mid = np.nonzero(edges > 0)[1].mean()
    #vis[:, int(mid)] = [0, 0, 255]

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    for i in xrange(len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("original", cv_image)
    cv2.imshow("mid", vis)

    #gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    #blur_gray = cv2.GaussianBlur(gray,(5, 5), 0)
    #edge_img = cv2.Canny(np.uint8(blur_gray), 60, 70)

    #cv2.imshow("original", cv_image)
    #cv2.imshow("gray", gray)
    #cv2.imshow("gaussian blur", blur_gray)
    #cv2.imshow("edge", edge_img)
    cv2.waitKey(1)

