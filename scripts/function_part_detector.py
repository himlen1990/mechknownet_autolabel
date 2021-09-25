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


class finger_detector:

  def __init__(self):
    self.bridge = CvBridge()
    self.cam_info_sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/camera_info",CameraInfo,self.cam_info_callback)
    self.rgb_sub = message_filters.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color", Image)
    self.depth_sub = message_filters.Subscriber("/hsrb/head_rgbd_sensor/depth_registered/image", Image)
    self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1)
    self.ts.registerCallback(self.callback)
    self.fingertip_pub = rospy.Publisher("/mechknownet/fingertip",PointStamped,queue_size=1)
    self.signal_pub = rospy.Publisher("/mechknownet/control_signal",String,queue_size=1)

    device = torch.device('cpu')
    self.model = SRHandNet(threshold=0.5)
    self.model.eval()
    self.model.to(device)

    self.fingertip_prepos = Point(-1000,-1000,-1000)
    self.holding_hand_pos = Point(-1000,-1000,-1000)
    self.timer_start = 0
    self.crop_object_flag = False
    self.pre_index_finger_point = (-1,-1)


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
        finger_point = (int(keypoint[3][1]),int(keypoint[3][0]))
        cv2.circle(my_frame,finger_point,3,(0,255,0),3)
    cv2.imshow("123",my_frame)      

    if not self.crop_object_flag and len(keypoints)> 10 and len(handrect)==1: #for handheld object detection
      all_z = []
      for keypoint in keypoints:
        finger_point = (int(keypoint[3][1]),int(keypoint[3][0]))
        z = depth_image[finger_point[1]][finger_point[0]]
        all_z.append(z)
      np.sort(all_z)
      mid_z = all_z[int(len(all_z)/2)]
      print "mid",mid_z

      # format of a keypoint is [number of hands(start from 0), joint, probability, (x,y)]
      for keypoint in keypoints:        
        if keypoint[1] == 7: #index fingertip  
          finger_point = [int(keypoint[3][1]),int(keypoint[3][0])]
          if not math.isnan(mid_z):
            nx,ny = self.unproject(finger_point[0],finger_point[1],mid_z)
            msg = PointStamped()
            msg.header = depth.header
            msg.point = Point(nx,ny,mid_z)
            self.fingertip_pub.publish(msg)
            
            if self.fingertip_prepos.z == -1000:
              self.fingertip_prepos = Point(nx,ny,mid_z)
            else:
              dis = abs(mid_z-self.fingertip_prepos.z)
              dis2 = math.sqrt(abs(nx-self.fingertip_prepos.x)**2 + abs(ny-self.fingertip_prepos.y)**2 + abs(mid_z-self.fingertip_prepos.z)**2)
              print "dis2:", dis2
              if dis2 > 0.03:
                self.fingertip_prepos = Point(nx,ny,mid_z)
              if dis2 < 0.018 and self.timer_start==0:
                self.timer_start = rospy.Time.now().to_sec()
              if dis2 < 0.018 and rospy.Time.now().to_sec()-self.timer_start>3.0:
                signal_str = String("get_object")
                self.signal_pub.publish(signal_str)
                self.holding_hand_pos = msg.point
                self.crop_object_flag = True
                print "stable!!!!!!!!!"
                self.timer_start = 0 #reset timer

    if self.crop_object_flag and len(handrect)==2 and len(keypoints)> 20: #point out function part
      all_z = []
      for keypoint in keypoints:
        if keypoint[0] == 1:
          finger_point = [int(keypoint[3][1]),int(keypoint[3][0])]
          if finger_point[1] > 479:
            finger_point[1] = 479
          if finger_point[0] > 479:
            finger_point[0] = 479

          z = depth_image[finger_point[1]][finger_point[0]]
          all_z.append(z)

      np.sort(all_z)
      mid_z = all_z[int(len(all_z)/2)]

      for keypoint in keypoints:
        if keypoint[0] == 1:
          if keypoint[1] == 7 or keypoint[1] == 6: #index fingertip
            finger_point = [int(keypoint[3][1]),int(keypoint[3][0])]
            if not math.isnan(mid_z):
              nx,ny = self.unproject(finger_point[0],finger_point[1],mid_z)
              dis_to_holding_hand = math.sqrt(abs(nx-self.holding_hand_pos.x)**2 + abs(ny-self.holding_hand_pos.y)**2 + abs(mid_z-self.holding_hand_pos.z)**2)
              if dis_to_holding_hand>0.05:
                print "!!!!!!!"
                print "2hand"
                print dis_to_holding_hand
                msg = PointStamped()
                msg.header = depth.header
                msg.header.stamp = rospy.Time.now()
                msg.point = Point(nx,ny,mid_z)
                self.fingertip_pub.publish(msg)

                if keypoint[1] == 7:
                  print finger_point
                  if self.pre_index_finger_point[0] == -1:
                    self.pre_index_finger_point = finger_point
                  if self.pre_index_finger_point[0] != -1:
                    changes = abs(finger_point[0]-self.pre_index_finger_point[0]) + abs(finger_point[1]-self.pre_index_finger_point[1])
                    self.pre_index_finger_point = finger_point
                    print changes
                    if self.timer_start == 0:
                      self.timer_start = rospy.Time.now().to_sec()
                    if changes > 5:
                      self.timer_start = rospy.Time.now().to_sec()
                    if changes < 5 and rospy.Time.now().to_sec()-self.timer_start>3.0:
                      signal_str = String("finish")
                      self.signal_pub.publish(signal_str)
                      print "finish"



    if self.crop_object_flag and len(handrect)==1 and len(keypoints)> 10: #point out function part
      all_z = []
      for keypoint in keypoints:
        finger_point = [int(keypoint[3][1]),int(keypoint[3][0])]
        if finger_point[1] > 479:
          finger_point[1] = 479
        if finger_point[0] > 479:
          finger_point[0] = 479

        z = depth_image[finger_point[1]][finger_point[0]]
        all_z.append(z)

      np.sort(all_z)
      mid_z = all_z[int(len(all_z)/2)]

      for keypoint in keypoints:
        if keypoint[1] == 7 or keypoint[1] == 6: #index fingertip
          finger_point = (int(keypoint[3][1]),int(keypoint[3][0]))
          if not math.isnan(mid_z):
            nx,ny = self.unproject(finger_point[0],finger_point[1],mid_z)
            dis_to_holding_hand = math.sqrt(abs(nx-self.holding_hand_pos.x)**2 + abs(ny-self.holding_hand_pos.y)**2 + abs(mid_z-self.holding_hand_pos.z)**2)

            if dis_to_holding_hand>0.05:
              print "send"
              print dis_to_holding_hand
              msg = PointStamped()
              msg.header = depth.header
              msg.header.stamp = rospy.Time.now()
              msg.point = Point(nx,ny,mid_z)
              self.fingertip_pub.publish(msg)
              

              if keypoint[1] == 7:
                if self.pre_index_finger_point[0] == -1:
                  self.pre_index_finger_point = finger_point
                if self.pre_index_finger_point[0] != -1:
                  changes = abs(finger_point[0]-self.pre_index_finger_point[0]) + abs(finger_point[1]-self.pre_index_finger_point[1])
                  self.pre_index_finger_point = finger_point
                  print changes
                  if self.timer_start == 0:
                    self.timer_start = rospy.Time.now().to_sec()
                  if changes > 5:
                    self.timer_start = rospy.Time.now().to_sec()
                  if changes < 5 and rospy.Time.now().to_sec()-self.timer_start>3.0:
                    signal_str = String("finish")
                    self.signal_pub.publish(signal_str)
                    print "finish"
                  

    k = cv2.waitKey(3) & 0xFF      
    #cv2.imshow("Image window", rgb_image)



def main(args):
  rospy.init_node('image_converter', anonymous=True)
  fd = finger_detector()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
