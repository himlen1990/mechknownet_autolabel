#!/usr/bin/env python

import roslib
roslib.load_manifest('mechknownet_autolabel')
import sys
import rospy
import cv2
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped,Point,Quaternion,PoseStamped,Pose
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
    self.grasp_pose_pub = rospy.Publisher("/mechknownet/grasp_pose",PoseStamped,queue_size=1)
    self.signal_pub = rospy.Publisher("/mechknownet/control_signal",String,queue_size=1)

    device = torch.device('cpu')
    self.model = SRHandNet(threshold=0.5)
    self.model.eval()
    self.model.to(device)

    self.fingertip_prepos = Point(-1000,-1000,-1000)
    self.holding_hand_pos = Point(-1000,-1000,-1000)
    self.timer_start = 0
    self.get_grasp_flag = False
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
      
  def project_point_to_vec(self,p1,p2,p3):
    v_p1p2 = p2-p1
    v_p1p3 = p3-p1
    dis_p1p2_square = np.sum(v_p1p2**2)
    v_p1o = np.sum(v_p1p2*v_p1p3)/(dis_p1p2_square)*v_p1p2
    o = p1 + v_p1o
    return o

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

    depth_for_show_1 = -100
    depth_for_show_2 = -100
    depth_for_show_3 = -100

    if not self.get_grasp_flag and len(keypoints)> 10 and len(handrect)==1: #for handheld object detection
      thumb = np.zeros(3)
      index_keypoint1 = np.zeros(3)
      index_keypoint2 = np.zeros(3)
      get_thumb_flag = False
      get_indexf1_flag = False
      get_indexf2_flag = False
      for keypoint in keypoints:        
        if keypoint[1] == 3: #thumb 
          finger_point = (int(keypoint[3][1]),int(keypoint[3][0]))
          z = depth_image[finger_point[1]][finger_point[0]]
          depth_for_show_1 = z
          if not math.isnan(z):
            nx,ny = self.unproject(finger_point[0],finger_point[1],z)
            thumb = np.array([nx,ny,z])
            get_thumb_flag = True

        if keypoint[1] == 6: #index keypt1
          finger_point = (int(keypoint[3][1]),int(keypoint[3][0]))
          z = depth_image[finger_point[1]][finger_point[0]]
          depth_for_show_2 = z
          #print "keypoint6 depth", z
          if not math.isnan(z):
            nx,ny = self.unproject(finger_point[0],finger_point[1],z)
            index_keypoint1 = np.array([nx,ny,z])
            get_indexf1_flag = True
            
        if keypoint[1] == 7: #index keypt2
          finger_point = (int(keypoint[3][1]),int(keypoint[3][0]))
          z = depth_image[finger_point[1]][finger_point[0]]
          depth_for_show_3 = z
          #print "keypoint7 depth", z
          if not math.isnan(z):
            nx,ny = self.unproject(finger_point[0],finger_point[1],z)
            index_keypoint2 = np.array([nx,ny,z])
            get_indexf2_flag = True
          
      #print "thumb ",thumb
      #print "index_keypoint1 ",index_keypoint1
      #print "index_keypoint2 ",index_keypoint2
      if get_thumb_flag and get_indexf1_flag and get_indexf2_flag:
        po = self.project_point_to_vec(index_keypoint1,index_keypoint2,thumb)      
        centroid = (thumb + po)/2
        #print centroid
        z_direction = index_keypoint2 - index_keypoint1
        z_direction_norm = z_direction / np.linalg.norm(z_direction)
        y_direction =      thumb -centroid 
        y_direction_norm = y_direction / np.linalg.norm(y_direction)
        x_direction = np.cross(z_direction,y_direction)
        x_direction_norm = z_direction / np.linalg.norm(z_direction)

        rotation_matrix = R.from_dcm([[x_direction_norm[0],x_direction_norm[1],x_direction_norm[2]],[y_direction_norm[0],y_direction_norm[1],y_direction_norm[2]],[z_direction_norm[0],z_direction_norm[1],z_direction_norm[2]]]) 
        #rotation_matrix = R.from_dcm([[z_direction_norm[0],z_direction_norm[1],z_direction_norm[2]],[x_direction_norm[0],x_direction_norm[1],x_direction_norm[2]],[y_direction_norm[0],y_direction_norm[1],y_direction_norm[2]]]) 

        quat = rotation_matrix.as_quat()
        msg = PoseStamped()
        msg.header = depth.header
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = Point(centroid[0],centroid[1],centroid[2])
        msg.pose.orientation = Quaternion(quat[0],quat[1],quat[2],quat[3]) #x, y, z, w
        self.grasp_pose_pub.publish(msg)


    for keypoint in keypoints:
        if keypoint[1] == 3: #thumb 
          finger_point = (int(keypoint[3][1]),int(keypoint[3][0]))
          cv2.circle(my_frame,finger_point,3,(255,0,0),3)
        if keypoint[1] == 6:
          finger_point = (int(keypoint[3][1]),int(keypoint[3][0]))
          cv2.circle(my_frame,finger_point,3,(0,255,0),3)
        if keypoint[1] == 7:
          finger_point = (int(keypoint[3][1]),int(keypoint[3][0]))
          cv2.circle(my_frame,finger_point,3,(0,0,255),3)
          if depth_for_show_1!=-100 and depth_for_show_2!=-100 and depth_for_show_3!=-100:
            text =  "keypoint3 depth: " + str(depth_for_show_1.tolist())
            text2 = "keypoint6 depth: " + str(depth_for_show_2.tolist())
            text3 = "keypoint7 depth: " + str(depth_for_show_3.tolist())
            cv2.putText(my_frame, text, (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            cv2.putText(my_frame, text2, (20, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv2.putText(my_frame, text3, (20, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    cv2.imshow("output",my_frame)      
        
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
