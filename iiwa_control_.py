#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:43:06 2021

@author: yuanbi
"""
import rospy
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_matrix
from iiwa_msgs.msg import CartesianPose
from geometry_msgs.msg import PoseStamped
#from iiwa_msgs.msg import WrenchStamped
from iiwa_msgs.msg import ControlMode
from iiwa_msgs.srv import ConfigureControlMode
from iiwa_msgs.msg import CartesianImpedanceControlMode
# from image_processing.msg import Reward_Pose
#from geometry_msgs.msg import PointStamped
#from control_msgs.msg import JointTrajectoryControllerState
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import math
import numpy as np
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys, os
import copy
import collections
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import PolynomialFeatures

# sys.path.append(os.path.expanduser("~/Projects/Yuan/Gaze guided RUSS/UltrasondConfienceMap-master/python"))
# from confidence_map import confidence_map2d

def rotation_mat_to_quaternion(rotaion_matrix):
    tr=rotaion_matrix[0,0]+rotaion_matrix[1,1]+rotaion_matrix[2,2]
    
    if tr>0:
        s=math.sqrt(tr+1)*2
        qw=0.25*s
        qx=(rotaion_matrix[2,1]-rotaion_matrix[1,2])/s
        qy=(rotaion_matrix[0,2]-rotaion_matrix[2,0])/s
        qz=(rotaion_matrix[1,0]-rotaion_matrix[0,1])/s
    elif rotaion_matrix[0,0]>rotaion_matrix[1,1] and rotaion_matrix[0,0]>rotaion_matrix[2,2]:
        s=math.sqrt(1+rotaion_matrix[0,0]-rotaion_matrix[1,1]-rotaion_matrix[2,2])*2
        qw=(rotaion_matrix[2,1]-rotaion_matrix[1,2])/s
        qx=0.25*s
        qy=(rotaion_matrix[0,1]+rotaion_matrix[1,0])/s
        qz=(rotaion_matrix[0,2]+rotaion_matrix[2,0])/s
    elif rotaion_matrix[1,1]>rotaion_matrix[2,2]:
        s=math.sqrt(1-rotaion_matrix[0,0]+rotaion_matrix[1,1]-rotaion_matrix[2,2])*2
        qw=(rotaion_matrix[0,2]-rotaion_matrix[2,0])/s
        qx=(rotaion_matrix[0,1]+rotaion_matrix[1,0])/s
        qy=0.25*s
        qz=(rotaion_matrix[1,2]+rotaion_matrix[2,1])/s
    else:
        s=math.sqrt(1-rotaion_matrix[0,0]-rotaion_matrix[1,1]+rotaion_matrix[2,2])
        qw=(rotaion_matrix[1,0]-rotaion_matrix[0,1])/s
        qx=(rotaion_matrix[0,2]+rotaion_matrix[2,0])/s
        qy=(rotaion_matrix[1,2]+rotaion_matrix[2,1])/s
        qz=0.25*s
    return qx,qy,qz,qw

# def rot_z(rad):
#     rotation_matrix=[[math.cos(rad),-math.sin(rad),0],[math.sin(rad),math.cos(rad),0],[0,0,1]]
#     rotation_matrix=np.dot(np.array([[-1,0,0],[0,1,0],[0,0,-1]]),np.array(rotation_matrix))
# #     rotation_matrix=np.dot(np.delete(quaternion_matrix(self.start_pos[3:7]),-1,1),np.array(rotation_matrix))
#     return np.array(rotation_matrix)


class iiwa_control():
    def __init__(self,probe_width=0.0513,image_depth=0.065,start_pos=np.array([0.4,0.3,0.12,1,0,0,0])):
        rospy.init_node("iiwa_control")
        self.command_pub=rospy.Publisher('/iiwa/command/CartesianPoseLin', PoseStamped, queue_size=10)
        self.pub_conf_img = rospy.Publisher("confImg",Image, queue_size=10)
        self.pub_conf_c_img = rospy.Publisher("confcImg",Image, queue_size=10)
#        rospy.Subscriber('/iiwa/PositionJointInterface_trajectory_controller/state',
#                         JointTrajectoryControllerState, self.jointStateCallback)
        rospy.Subscriber('/iiwa/state/CartesianPose',CartesianPose,self.updatePose)
        # rospy.Subscriber('/iiwa/state/CartesianPose',PoseStamped,self.updatePose)
#        rospy.Subscriber('/iiwa/state/CartesianWrench',WrenchStamped,self.updateWrench)
        rospy.Subscriber("USImg",Image,self.updateImage)
        rospy.Subscriber("segmentedImg",Image,self.updateSegmentImage)
        # rospy.Subscriber("confImg",Image,self.updateConfImage)
        # rospy.set_param('/iiwa/toolName', 'cephalinear')
        self.robotStopped=True;
#        self.actualJS=np.array([0, 0, 0, 0, 0, 0, 0]);
#        self.desiredJS=np.array([0, 0, 0, 0, 0, 0, 0]);
        self.probe_width=probe_width # m
        self.image_depth=image_depth # m

        self.segCen_rec=collections.deque(maxlen=40)
        self.current_pos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.current_rot_matrix=None
        self.desired_rot_matrix=None
        self.force=np.array([0.0, 0.0, 0.0])
        self.torque=np.array([0.0, 0.0, 0.0])
        self.start_pos=start_pos
        self.image = None
        self.SegImage = None
        self.confidence = None
        self.bridge = CvBridge()
        self.client_config=rospy.ServiceProxy('/iiwa/configuration/ConfigureControlMode',ConfigureControlMode)
        self.best_reward=0
        self.current_reward=None
        print('initialised')
    def move_to_cartesian_pose(self,desiredPose,z_needed,precision=1,check=True):
        posemsg=PoseStamped()
        posemsg.header.frame_id = "iiwa_link_0";
    
        posemsg.pose.position.x = desiredPose[0]
        posemsg.pose.position.y = desiredPose[1]
        posemsg.pose.position.z = desiredPose[2]
        posemsg.pose.orientation.x = desiredPose[3]
        posemsg.pose.orientation.y = desiredPose[4]
        posemsg.pose.orientation.z = desiredPose[5]
        posemsg.pose.orientation.w = desiredPose[6]
        
        self.desired_rot_matrix= quaternion_matrix(desiredPose[3:7])
        
        self.command_pub.publish(posemsg)
        if check:
            start_time=time.time()
            self.robotStopped=False
            while not self.robotStopped:

                self.current_rot_matrix= quaternion_matrix(self.current_pos[3:7])
                if z_needed:
                    pos_reached=bool(np.sum(np.abs(self.current_pos[0:3]-np.array(desiredPose)[0:3]))<0.005*precision)
                else:
                    pos_reached=bool(np.sum(np.abs(self.current_pos[0:2]-np.array(desiredPose)[0:2]))<0.005*precision)
                rot_reached=bool(np.sum(np.abs(np.delete(np.array(self.current_rot_matrix),-1,1)-np.delete(np.array(self.desired_rot_matrix),-1,1)))<0.1*precision)

                self.robotStopped=(pos_reached and rot_reached) or time.time()-start_time>10
    #             sys.stdout.write('pos: %s rot: %s \r' % (pos_reached,rot_reached))
    #             sys.stdout.flush()
    #             print('pos: ',np.sum(np.abs(self.current_pos[0:3]-np.array(desiredPose)[0:3])))
    #             print('rot: ',rot_reached)
    #             print('robot stopped: ',pos_reached and rot_reached)
    #             print('rot_current: ',np.delete(np.array(self.current_rot_matrix),0,1))
    #             print('rot_current: ',np.delete(np.array(self.desired_rot_matrix),0,1))
                self.command_pub.publish(posemsg)
    #            print(np.sum(np.abs(self.current_pos-np.array(desiredPose))))
    #            print(time.time()-start_time)
        time.sleep(1)
    def jointStateCallback(self,msg):
        self.actualJS=np.array(msg.actual.positions)
        self.desiredJS=np.array(msg.desired.positions)
        
    def move_to(self,pose,precision=2,check=True):
        rotaion_matrix=self.rot_z(pose[2])
        qx,qy,qz,qw=rotation_mat_to_quaternion(rotaion_matrix)
        desiredPose=[self.start_pos[0]+pose[0], self.start_pos[1]+pose[1], self.start_pos[2], qx,qy,qz,qw]
#         print(desiredPose)
        self.move_to_cartesian_pose(desiredPose,False,precision,check=check)
        
    def move_to_start(self,check=True):
        self.move_to_cartesian_pose(self.start_pos,True,check=check)
        print('move to start finished')
        
    def updatePose(self,msg):
        self.current_pos[0]=msg.poseStamped.pose.position.x
        self.current_pos[1]=msg.poseStamped.pose.position.y
        self.current_pos[2]=msg.poseStamped.pose.position.z
        self.current_pos[3]=msg.poseStamped.pose.orientation.x
        self.current_pos[4]=msg.poseStamped.pose.orientation.y
        self.current_pos[5]=msg.poseStamped.pose.orientation.z
        self.current_pos[6]=msg.poseStamped.pose.orientation.w
        
        
        # self.current_pos[0]=msg.pose.position.x
        # self.current_pos[1]=msg.pose.position.y
        # self.current_pos[2]=msg.pose.position.z
        # self.current_pos[3]=msg.pose.orientation.x
        # self.current_pos[4]=msg.pose.orientation.y
        # self.current_pos[5]=msg.pose.orientation.z
        # self.current_pos[6]=msg.pose.orientation.w
        
        self.euler = euler_from_quaternion(self.current_pos[3:])
        
        # if not self.robotStopped:
        #     self.segCen_rec.append(self.current_pos[:3])
        
#     def updateWrench(self,msg):
#         self.force[0]=msg.wrench.force.x
#         self.force[1]=msg.wrench.force.y
#         self.force[2]=msg.wrench.force.z
#         self.torque[0]=msg.wrench.torque.x
#         self.torque[1]=msg.wrench.torque.x
#         self.torque[2]=msg.wrench.torque.x
        
    def updateImage(self,msg):
        msg.encoding = 'mono8'
        tmp = cv2.resize(self.bridge.imgmsg_to_cv2(msg),(256,256),interpolation=cv2.INTER_LANCZOS4)
        self.image=tmp.astype('float')/255
        
    # def updateConfImage(self,msg):
    #     msg.encoding = 'mono8'
    #     tmp = cv2.resize(self.bridge.imgmsg_to_cv2(msg),(256,256),interpolation=cv2.INTER_LANCZOS4)
    #     self.confidence=tmp.astype('float')/255
        
    def controller_init(self):
        msg_config=ControlMode()
        control_mode=msg_config.CARTESIAN_IMPEDANCE
        impedance_config=CartesianImpedanceControlMode()
        
        impedance_config.cartesian_stiffness.x=1200
        impedance_config.cartesian_stiffness.y=1200
        impedance_config.cartesian_stiffness.z=400
        impedance_config.cartesian_stiffness.a=150
        impedance_config.cartesian_stiffness.b=150
        impedance_config.cartesian_stiffness.c=150
        
        impedance_config.cartesian_damping.x=0.8
        impedance_config.cartesian_damping.y=0.8
        impedance_config.cartesian_damping.z=0.8
        impedance_config.cartesian_damping.a=0.8
        impedance_config.cartesian_damping.b=0.8
        impedance_config.cartesian_damping.c=0.8
        
        impedance_config.nullspace_stiffness=200
        impedance_config.nullspace_damping=1.0
        
        try:
            self.client_config(control_mode,None,impedance_config,None,None,None)
            print('cartesian impedance control mode activated')
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
            rospy.shutdown()
            
    def rot_z(self,rad):
        rotation_matrix=[[math.cos(rad),-math.sin(rad),0],[math.sin(rad),math.cos(rad),0],[0,0,1]]
    #     rotation_matrix=np.dot(np.array([[-1,0,0],[0,1,0],[0,0,-1]]),np.array(rotation_matrix))
        rotation_matrix=np.dot(np.delete(quaternion_matrix(self.start_pos[3:7]),-1,1),np.array(rotation_matrix))
        return np.array(rotation_matrix)
    
    def updateSegmentImage(self,msg):
        msg.encoding = 'mono8'
        tmp = cv2.resize(self.bridge.imgmsg_to_cv2(msg),(256,256),interpolation=cv2.INTER_LANCZOS4)
        self.SegImage=tmp.astype('float')/255
        # if not self.robotStopped:
        if self.SegImage is not None:
            ret,thresh = cv2.threshold(self.SegImage,0.5,1,0)
            contours = cv2.findContours(thresh.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            if len(contours)>0:
                big_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(big_contour)
                if M["m00"]>0:
                    cx_img = int(M["m10"] / M["m00"])
                    cy_img = int(M["m01"] / M["m00"])
                    self.c_end = [(cx_img-127)*self.probe_width/256, 0.0, cy_img*self.image_depth/256, 1.0]
                    R_end_base = quaternion_matrix(self.current_pos[3:7])
                    R_end_base[0:3,3] = self.current_pos[0:3]
                    self.c_base = np.dot(R_end_base,self.c_end)
                    self.segCen_rec.append(self.c_base)

    def centerline_correction(self,vec=None,check=True):
        if vec is None:
            segCen_rec_arr = np.array(self.segCen_rec)
            x_line=segCen_rec_arr[:,0]
            y_line=segCen_rec_arr[:,1]
            degree = 3
            coefficients = np.polyfit(x_line, y_line, degree)
            derivative_coefficients = np.polyder(coefficients)
            derivative_polynomial = np.poly1d(derivative_coefficients)
            last_x = x_line[-1]
            gradient_at_last_point = derivative_polynomial(last_x)

            vec = np.array([1, gradient_at_last_point, 0])
        vec = vec/np.linalg.norm(vec) 
        R_e = quaternion_matrix(self.current_pos[3:7])[0:3,0:3]
        y_e = R_e[0:3,1]
        # y_e = np.array([0, 1, 0])

        angle = np.arccos(np.dot(y_e,vec)/(np.linalg.norm(y_e)*np.linalg.norm(vec)))
        if angle>np.pi/2:
            axis = np.cross(y_e, -vec)
            angle = np.arccos(np.dot(y_e,-vec)/(np.linalg.norm(y_e)*np.linalg.norm(-vec)))
        else:
            axis = np.cross(y_e, vec)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R_cl = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        # R_new = np.dot(R,R_e)
        
        # qx,qy,qz,qw=rotation_mat_to_quaternion(R_new)
        # desiredPose=[self.current_pos[0], self.current_pos[1], self.current_pos[2], qx,qy,qz,qw]
        # self.move_to_cartesian_pose(desiredPose,True,check=check)
        return R_cl
        
    def confidence_correction(self,r=0.1,sl=True):
        if sl:
            confidence = self.get_confidence_sl(self.image)
        else:
            confidence = self.get_confidence(self.image)
        
        rows, cols = np.indices(confidence.shape)
        weighted_row = np.sum(rows * confidence,axis=0)
        weighted_col = np.sum(cols[0] * weighted_row) / np.sum(weighted_row)
        w_c = (weighted_col-confidence.shape[1]//2)*(self.probe_width/confidence.shape[1])
        # self.send_conf_image(confidence)
        # self.send_conf_c_image(confidence, weighted_col)
        theta = np.arctan(w_c/r)
        # self.rot_y_(-theta)
        return theta
        
    def get_confidence(self,img):
        
        conf_map = confidence_map2d(img, alpha=1.5, beta=90, gamma=0.03, spacing=None, solver_mode='bf')
        return (conf_map*255).astype('uint8')
    
    def get_confidence_sl(self,img):
        RF = img**2
        
        sum_c_RF = np.sum(RF, axis=0)
        ac_sum_c_RF = np.cumsum(RF, axis=0)
        conf = (1-ac_sum_c_RF/sum_c_RF)*255
        
        return conf.astype('uint8')
        
    def send_conf_image(self,conf):
        msg = self.bridge.cv2_to_imgmsg(conf, encoding="mono8")
        self.pub_conf_img.publish(msg)
        
    def send_conf_c_image(self,conf,w_c):
        start_point = (conf.shape[1]//2, 0)
        end_point = (conf.shape[1]//2, int(conf.shape[0]))
        start_point_c = (int(w_c), 0)
        end_point_c = (int(w_c), int(conf.shape[0]))
        
        
        # Define the color of the line in grayscale (255 for white, 0 for black)
        color = 255  # White in this example
        confidence_c = copy.deepcopy(conf)
        # Draw the line on the grayscale image
        thickness = 2  # in pixels
        confidence_c=cv2.line(confidence_c, start_point, end_point, color, thickness)
        confidence_c=cv2.line(confidence_c, start_point_c, end_point_c, color, thickness)
        msg = self.bridge.cv2_to_imgmsg(confidence_c, encoding="mono8")
        self.pub_conf_c_img.publish(msg)
        
    def rot_x_(self,rad,check=True):
        rotation_matrix=[[1,0,0],[0,math.cos(rad),-math.sin(rad)],[0,math.sin(rad),math.cos(rad)]]
        current_pos = copy.deepcopy(self.current_pos)
        
        rotation_matrix=np.dot(quaternion_matrix(current_pos[3:7])[0:3,0:3], np.array(rotation_matrix))
        
        qx,qy,qz,qw=rotation_mat_to_quaternion(rotation_matrix)
        desiredPose=[current_pos[0], current_pos[1], current_pos[2], qx,qy,qz,qw]
        self.move_to_cartesian_pose(desiredPose,True,check=check)
        return rotation_matrix
        
    def rot_y_(self,rad,check=True):
        rotation_matrix=[[math.cos(rad),0,math.sin(rad)],[0,1,0],[-math.sin(rad),0,math.cos(rad)]]
        
        rotation_matrix=np.dot(quaternion_matrix(self.current_pos[3:7])[0:3,0:3], np.array(rotation_matrix))
        
        qx,qy,qz,qw=rotation_mat_to_quaternion(rotation_matrix)
        desiredPose=[self.current_pos[0], self.current_pos[1], self.current_pos[2], qx,qy,qz,qw]
        self.move_to_cartesian_pose(desiredPose,True,check=check)
        return rotation_matrix
        
    def rot_z_(self,rad,check=True):
        rotation_matrix=[[math.cos(rad),-math.sin(rad),0],[math.sin(rad),math.cos(rad),0],[0,0,1]]
        rotation_matrix=np.dot(quaternion_matrix(self.current_pos[3:7])[0:3,0:3], np.array(rotation_matrix))
        
        qx,qy,qz,qw=rotation_mat_to_quaternion(rotation_matrix)
        desiredPose=[self.current_pos[0], self.current_pos[1], self.current_pos[2], qx,qy,qz,qw]
        self.move_to_cartesian_pose(desiredPose,True,check=check)
        return rotation_matrix
        
    def step_move(self,pose,precision=1,check=True):
#         pose = [x,y,z,rot_x,rot_y,rot_z] (m,rad)
        
        pos_e = np.array([pose[0], pose[1], pose[2], 1])
        R_end_base = quaternion_matrix(self.current_pos[3:7])
        R_end_base[0:3,3] = self.current_pos[0:3] 
        pos_b = np.dot(R_end_base,pos_e)
        target_pose = self.current_pos
        target_pose[0:3] = pos_b[0:3]
        self.move_to_cartesian_pose(target_pose,True,precision,check=check)
        if pose[3]!=0:
            self.rot_x_(pose[3])
        if pose[4]!=0:
            self.rot_y_(pose[4])
        if pose[5]!=0:
            self.rot_z_(pose[5])
            
    def get_rotation_matrix(self):
        R = quaternion_matrix(self.current_pos[3:7])
        R[:3,3] = self.current_pos[0:3]
        return R
    
    def vessel_correction(self):
        pose = np.zeros(6)
        pose[0] = self.c_end[0]
        self.step_move(pose)
        
    def step_control(self,pose,precision=1):
#         pose = [x,y,z,rot_x,rot_y,rot_z] (m,rad)
        
        pos_e = np.array([pose[0], pose[1], pose[2], 1])
        R_end_base = quaternion_matrix(self.current_pos[3:7])
        R_end_base[0:3,3] = self.current_pos[0:3] 
        pos_b = np.dot(R_end_base,pos_e)
        target_pose = self.current_pos
        target_pose[0:3] = pos_b[0:3]
        self.move_to_cartesian_pose(target_pose,True,precision,False)
        if pose[3]!=0:
            self.rot_x_(pose[3])
        if pose[4]!=0:
            self.rot_y_(pose[4])
        if pose[5]!=0:
            self.rot_z_(pose[5])
            
    def save_home(self):
        self.home_pose = copy.deepcopy(self.current_pos)
        print('home position saved')
        
    def save_out(self):
        self.out_pose = copy.deepcopy(self.current_pos)
        print('out position saved')
        
    def move_home(self):
        self.move_to_cartesian_pose(self.home_pose, z_needed=True)
        
    def move_out(self):
        self.move_to_cartesian_pose(self.out_pose, z_needed=True)
        
    def move_out_control(self, step=0.01,centerline_correction=False, vessel_correction=False, confidence_correction=False, r=0.1, sl=True,precision=1):
        times = [0, 1]
        distance = np.linalg.norm(self.home_pose[:3] - self.out_pose[:3])
        steps = int(distance / step) 
        
        positions = np.linspace(self.home_pose[:3], self.out_pose[:3], steps + 1)
        
        rotations = R.from_quat([self.home_pose[3:], self.out_pose[3:]])
        
        slerp = Slerp(times, rotations)
        interpolation_times = np.linspace(0, 1, steps + 1)
        interpolated_rotations = slerp(interpolation_times)
        interpolated_quats = interpolated_rotations.as_quat()
        
        interpolated_poses = np.hstack((positions, interpolated_quats))
        interpolated_poses = np.array(interpolated_poses)
        
        for i in range(1,len(interpolated_poses)):
            # if centerline_correction:
            #     if i <2:
            #         self.move_to_cartesian_pose(interpolated_poses[i],z_needed=True,precision=1,check=True)
            #     else:
            #         R_cl = self.centerline_correction()
            #         rotation_matrix=np.dot(np.array(R_cl),quaternion_matrix(self.current_pos[3:7])[0:3,0:3])
                    
            #         qx,qy,qz,qw=rotation_mat_to_quaternion(rotation_matrix)
                    
            #         target_pose = interpolated_poses[i]
            #         target_pose[3:7] = [qx,qy,qz,qw]
                    
            #         self.move_to_cartesian_pose(target_pose,z_needed=True,precision=1,check=True)
                
            if vessel_correction and confidence_correction:
                # vec_forward = interpolated_poses[i][:3]-interpolated_poses[i-1][:3]
                
                pos_e = np.array([self.c_end[0], 0, 0, 1])
                R_end_base = quaternion_matrix(self.current_pos[3:7])
                R_end_base[:3,3] = self.current_pos[:3]
                pos_b = np.dot(R_end_base,pos_e)
                vec_vessel = pos_b[:3]-self.current_pos[:3]
                
                theta_confidence = self.confidence_correction(r=r)
                rotation_matrix=[[math.cos(theta_confidence),0,math.sin(theta_confidence)],[0,1,0],[-math.sin(theta_confidence),0,math.cos(theta_confidence)]]
                
                rotation_matrix=np.dot(quaternion_matrix(self.current_pos[3:7])[0:3,0:3], np.array(rotation_matrix))
                
                qx,qy,qz,qw=rotation_mat_to_quaternion(rotation_matrix)
                
                interpolated_poses[:,:3] = interpolated_poses[:,:3]+vec_vessel
                target_pose = interpolated_poses[i]
                target_pose[3:7] = [qx,qy,qz,qw]
                # target_pose[:3] = target_pose[:3]+vec_forward+vec_vessel
                self.move_to_cartesian_pose(target_pose,z_needed=True,precision=precision,check=True)
            elif vessel_correction:
                # vec_forward = interpolated_poses[i][:3]-interpolated_poses[i-1][:3]
                
                pos_e = np.array([self.c_end[0], 0, 0, 1])
                R_end_base = quaternion_matrix(self.current_pos[3:7])
                R_end_base[:3,3] = self.current_pos[:3]
                pos_b = np.dot(R_end_base,pos_e)
                vec_vessel = pos_b[:3]-self.current_pos[:3]
                
                interpolated_poses[:,:3] = interpolated_poses[:,:3]+vec_vessel
                print(interpolated_poses[i])
                self.move_to_cartesian_pose(interpolated_poses[i],z_needed=True,precision=precision,check=True)
                
            elif confidence_correction:
                theta_confidence = self.confidence_correction(r=r)
                rotation_matrix=[[math.cos(theta_confidence),0,math.sin(theta_confidence)],[0,1,0],[-math.sin(theta_confidence),0,math.cos(theta_confidence)]]
                
                rotation_matrix=np.dot(quaternion_matrix(self.current_pos[3:7])[0:3,0:3], np.array(rotation_matrix))
                
                qx,qy,qz,qw=rotation_mat_to_quaternion(rotation_matrix)
                
                target_pose = interpolated_poses[i]
                target_pose[3:7] = [qx,qy,qz,qw]
                
                self.move_to_cartesian_pose(target_pose,z_needed=True,precision=precision,check=True)
            
            else:
                self.move_to_cartesian_pose(interpolated_poses[i],z_needed=True,precision=precision,check=True)
        
        # for pose in interpolated_poses:
        #     if vessel_correction:
        #         pos_e = np.array([self.c_end[0], 0, 0, 1])
        #         R_end_base = quaternion_matrix(self.current_pos[3:7])
        #         R_end_base[0:3,3] = self.current_pos[0:3]
        #         pos_b = np.dot(R_end_base,pos_e)
        #         vec = pos_b[0:3]-self.current_pos[0:3]
        #         pose[0:3] = pose[0:3]+vec
        #         print(pose)
        #         # target_poses.append(target_pose)
        #         self.move_to_cartesian_pose(pose,z_needed=True,precision=1,check=True)
        #         # print(pose)
        #     else:
        #         # print(pose)
        #         self.move_to_carte1sian_pose(pose,z_needed=True,precision=1,check=True)
        
        return interpolated_poses
    
    def move_home_control(self, step=0.01, vessel_correction=False, confidence_correction=False, r=0.1, sl=True,precision=1):
        # target_poses=[]
        times = [0, 1]
        distance = np.linalg.norm(self.out_pose[:3] - self.home_pose[:3])
        steps = int(distance / step) 
        
        positions = np.linspace(self.out_pose[:3], self.home_pose[:3], steps + 1)
        
        rotations = R.from_quat([self.out_pose[3:], self.home_pose[3:]])
        
        slerp = Slerp(times, rotations)
        interpolation_times = np.linspace(0, 1, steps + 1)
        interpolated_rotations = slerp(interpolation_times)
        interpolated_quats = interpolated_rotations.as_quat()
        
        interpolated_poses = np.hstack((positions, interpolated_quats))
        
        for i in range(1,len(interpolated_poses)):
            if vessel_correction and confidence_correction:
                # vec_forward = interpolated_poses[i][:3]-interpolated_poses[i-1][:3]
                
                pos_e = np.array([self.c_end[0], 0, 0, 1])
                R_end_base = quaternion_matrix(self.current_pos[3:7])
                R_end_base[:3,3] = self.current_pos[:3]
                pos_b = np.dot(R_end_base,pos_e)
                vec_vessel = pos_b[:3]-self.current_pos[:3]
                
                theta_confidence = self.confidence_correction(r=r)
                rotation_matrix=[[math.cos(theta_confidence),0,math.sin(theta_confidence)],[0,1,0],[-math.sin(theta_confidence),0,math.cos(theta_confidence)]]
                
                rotation_matrix=np.dot(quaternion_matrix(self.current_pos[3:7])[0:3,0:3], np.array(rotation_matrix))
                
                qx,qy,qz,qw=rotation_mat_to_quaternion(rotation_matrix)
                
                interpolated_poses[:,:3] = interpolated_poses[:,:3]+vec_vessel
                target_pose = interpolated_poses[i]
                target_pose[3:7] = [qx,qy,qz,qw]
                # target_pose[:3] = target_pose[:3]+vec_forward+vec_vessel
                self.move_to_cartesian_pose(target_pose,z_needed=True,precision=precision,check=True)
            elif vessel_correction:
                # vec_forward = interpolated_poses[i][:3]-interpolated_poses[i-1][:3]
                
                pos_e = np.array([self.c_end[0], 0, 0, 1])
                R_end_base = quaternion_matrix(self.current_pos[3:7])
                R_end_base[:3,3] = self.current_pos[:3]
                pos_b = np.dot(R_end_base,pos_e)
                vec_vessel = pos_b[:3]-self.current_pos[:3]
                
                interpolated_poses[:,:3] = interpolated_poses[:,:3]+vec_vessel
                print(interpolated_poses[i])
                self.move_to_cartesian_pose(interpolated_poses[i],z_needed=True,precision=precision,check=True)
                
            elif confidence_correction:
                theta_confidence = self.confidence_correction(r=r)
                rotation_matrix=[[math.cos(theta_confidence),0,math.sin(theta_confidence)],[0,1,0],[-math.sin(theta_confidence),0,math.cos(theta_confidence)]]
                
                rotation_matrix=np.dot(quaternion_matrix(self.current_pos[3:7])[0:3,0:3], np.array(rotation_matrix))
                
                qx,qy,qz,qw=rotation_mat_to_quaternion(rotation_matrix)
                
                target_pose = interpolated_poses[i]
                target_pose[3:7] = [qx,qy,qz,qw]
                
                self.move_to_cartesian_pose(target_pose,z_needed=True,precision=precision,check=True)
            
            else:
                self.move_to_cartesian_pose(interpolated_poses[i],z_needed=True,precision=precision,check=True)
        
        return interpolated_poses
        
        # self.home_pose
        # pos_e = np.array([0, step, 0, 1])
        # R_end_base = quaternion_matrix(self.current_pos[3:7])
        # R_end_base[0:3,3] = self.current_pos[0:3] 
        # pos_b = np.dot(R_end_base,pos_e)
        # target_pose = self.current_pos
        # target_pose[0:3] = pos_b[0:3]
        # self.move_to_cartesian_pose(target_pose,z_needed=True,precision=1,check=True)
        
        
        
        
#iiwa=iiwa_control()
#iiwa.move_to_start()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        