#!/usr/bin/env python

import roslib
roslib.load_manifest('mechknownet_autolabel')
import sys
import rospy
import cv2
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped,Point
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import message_filters
import tf as ros_tf
import math

import argparse
from datetime import datetime
import os.path as osp

import torch
from eos import make_fancy_output_dir
from srhandnet import SRHandNet
from srhandnet.visualization import visualize_hand_keypoints
from srhandnet.visualization import visualize_hand_rects
from srhandnet.visualization import visualize_text


class trajectory_detector:

  def __init__(self):
    self.bridge = CvBridge()
    self.cam_info_sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/camera_info",CameraInfo,self.cam_info_callback)
    self.rgb_sub = message_filters.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color", Image)
    self.depth_sub = message_filters.Subscriber("/hsrb/head_rgbd_sensor/depth_registered/image", Image)
    self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1)
    self.ts.registerCallback(self.callback)
    self.hand_pub = rospy.Publisher("/mechknownet/hand_position",PointStamped,queue_size=5)
    self.signal_pub = rospy.Publisher("/mechknownet/control_signal",String,queue_size=1)

    device = torch.device('cpu')
    self.model = SRHandNet(threshold=0.5)
    self.model.eval()
    self.model.to(device)

    self.hand_prepos = Point(-1000,-1000,-1000)
    self.holding_hand_pos = Point(-1000,-1000,-1000)
    self.timer_start = 0
    self.crop_object_flag = False


  def cam_info_callback(self, msg):
    self.fx = msg.K[0]
    self.fy = msg.K[4]
    self.cx = msg.K[2]
    self.cy = msg.K[5]
    self.invfx = 1.0/self.fx
    self.invfy = 1.0/self.fy
    self.cam_info_sub.unregister()

  def unproject(self,u,v,z):
      x = (u - self.cx) * z * self.invfx
      y = (v - self.cy) * z * self.invfy
      return x,y


  def compute_hand_position(self,keypoints,depth_image):
      all_z = []
      all_x = []
      all_y = []
      for keypoint in keypoints:
        key_point_xy = (int(keypoint[3][1]),int(keypoint[3][0]))
        z = depth_image[key_point_xy[1]][key_point_xy[0]]
        if not math.isnan(z):
          nx,ny = self.unproject(key_point_xy[0],key_point_xy[1],z)
          all_z.append(z)
          all_x.append(nx)
          all_y.append(ny)
      all_z_np = np.array(all_z)
      all_x_np = np.array(all_x)
      all_y_np = np.array(all_y)
      rank = np.argsort(all_z)
      sorted_z = all_z_np[rank]
      sorted_x = all_x_np[rank]
      sorted_y = all_y_np[rank]

      index = int(len(all_z)/2)
      mid_z = sorted_z[index]
      mid_x = sorted_x[index]
      mid_y = sorted_y[index]
      return mid_x, mid_y, mid_z

  def callback(self,rgb, depth):
    try:
      rgb_image = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
    except CvBridgeError as e:
      print(e)

    try:
      depth_image = self.bridge.imgmsg_to_cv2(depth, "passthrough")
    except CvBridgeError as e:
      print(e)
    
    frame = rgb_image.copy()
    keypoints, handrect = self.model.pyramid_inference(
      frame, return_bbox=True)        

    my_frame = frame.copy()
    for keypoint in keypoints:
        key_point_xy = (int(keypoint[3][1]),int(keypoint[3][0]))
        cv2.circle(my_frame,key_point_xy,3,(255,0,0),3)
    cv2.imshow("123",my_frame)      

    if not self.crop_object_flag and len(keypoints)> 10 and len(handrect)==1: #wait for hand stable
      hand_x, hand_y, hand_z = self.compute_hand_position(keypoints,depth_image)
      msg = PointStamped()
      msg.header = depth.header
      msg.header.stamp = rospy.Time.now()
      msg.point = Point(hand_x,hand_y,hand_z)
      self.hand_pub.publish(msg)

      if self.hand_prepos.z == -1000:
        self.hand_prepos = Point(hand_x,hand_y,hand_z)
      else:
        dis = math.sqrt(abs(hand_x-self.hand_prepos.x)**2 + abs(hand_y-self.hand_prepos.y)**2 + abs(hand_z-self.hand_prepos.z)**2)
        print "dis ",dis
        if dis > 0.05:
          self.hand_prepos = Point(hand_x,hand_y,hand_z)
          self.timer_start = 0
        if dis < 0.03 and self.timer_start==0:
          self.timer_start = rospy.Time.now().to_sec()
        if dis < 0.03 and rospy.Time.now().to_sec()-self.timer_start>3.0:
          signal_str = String("hand_stable")
          self.signal_pub.publish(signal_str)
          #self.holding_hand_pos = msg.point
          self.crop_object_flag = True
          print "now!!!!!!!!!"
          

    if self.crop_object_flag and len(handrect)==1 and len(keypoints)> 10: #hand move
      hand_x, hand_y, hand_z = self.compute_hand_position(keypoints,depth_image)
      print "send"
      print hand_z
      msg = PointStamped()
      msg.header = depth.header
      msg.header.stamp = rospy.Time.now()
      msg.point = Point(hand_x, hand_y, hand_z)
      self.hand_pub.publish(msg)
      signal_str = String("hand_moving")
      self.signal_pub.publish(signal_str)




    k = cv2.waitKey(3) & 0xFF      
    #cv2.imshow("Image window", rgb_image)



def main(args):
  rospy.init_node('image_converter', anonymous=True)
  td = trajectory_detector()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
