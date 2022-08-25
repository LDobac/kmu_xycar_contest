#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from ar_track_alvar_msgs.msg import AlvarMarker, AlvarMarkers

def callback(marker):
    print(marker)

if "__main__" == __name__:
  rospy.init_node("my_alvar")
  image_sub = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, callback)

  while not rospy.is_shutdown():
      pass
