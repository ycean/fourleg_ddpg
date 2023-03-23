#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #
#MODIFICATION FOR QUADRUPED ROBOT TRAINING BY: YEOH CHIN EAN

import rospy
import time
import actionlib
import numpy as np
import math
from math import pi
from std_msgs.msg import String, Float64, UInt16
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryActionGoal, FollowJointTrajectoryGoal
from dynamixel_workbench_msgs.srv import JointCommand, JointCommandRequest, JointCommandResponse
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Twist, Transform, TransformStamped, Vector3
from sensor_msgs.msg import Temperature, Imu, JointState, LaserScan
from math import* 
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import random
from random import*
import copy
target_not_movable = False

CONV_ = 4096/(2*pi)
J1_MAX_ = -0.785
J1_MIN_ = -1.2
J1_MAX = 1.2 #radian
J2_MAX = 0.2094
J3_MAX = 0.1675 #direction of axis in gazebo and real robot is opposite so no -ve sign
J1_MIN = 0.785 #radian
J2_MIN = 0.1675
J3_MIN = 0.2094 #direction of axis in gazebo and real robot is opposite so no -ve sign

class Env():
    def __init__(self, action_dim=12):
    	#rospy.init_node('robot_environment_node')
    	self.goal_x = 0
    	self.goal_y = 0
    	self.pose_x = 0
    	self.pose_y = 0
    	self.pose_z = 0.139
    	self.heading = 0
    	self.initGoal = True
    	self.touch_goal = False
    	self.joint_state = JointState()
    	self.init_time = 0
    	self.delta_time = 0
    	# JOINT OUTPUT PUBLISH IS CHANGE TO JOINTCOMMAND THROUGH SERVICE AND CLIENT
    	rospy.wait_for_service('/joint_command')
    	self.joint_pub = rospy.ServiceProxy('/joint_command', JointCommand)
    	self.input_val = JointCommandRequest()
    	
    	self.joint_pos_max = np.array([0, 0, 0,
    					0, 0, 0,
    					0, 0, 0,
    					0, 0, 0])
    	self.joint_pos_max_F = np.array([J1_MAX_, J2_MAX, J3_MAX,
    					  J1_MAX, J2_MAX, J3_MAX,
    					  J1_MAX, J2_MAX, J3_MAX,
    					  J1_MAX_, J2_MAX, J3_MAX])
    	self.joint_pos_max_R = np.array([J1_MAX_, J2_MAX, J3_MAX,
    					  J1_MAX_, J2_MAX, J3_MAX,
    					  J1_MAX, J2_MAX, J3_MAX,
    					  J1_MAX, J2_MAX, J3_MAX])
    	self.joint_pos_max_B = np.array([ J1_MAX, J2_MAX, J3_MAX,
    					   J1_MAX_, J2_MAX, J3_MAX,
    					   J1_MAX_, J2_MAX, J3_MAX,
    					   J1_MAX, J2_MAX, J3_MAX])
    	self.joint_pos_max_L = np.array([ J1_MAX, J2_MAX, J3_MAX,
    					   J1_MAX, J2_MAX, J3_MAX,
    					   J1_MAX_, J2_MAX, J3_MAX,
    					   J1_MAX_, J2_MAX, J3_MAX])
    	
    	self.joint_pos_min_F = np.array([J1_MIN_, J2_MIN, J3_MIN,
    					  J1_MIN, J2_MIN, J3_MIN,
    					  J1_MIN, J2_MIN, J3_MIN,
    					  J1_MIN_, J2_MIN, J3_MIN])
    	self.joint_pos_min_R = np.array([J1_MIN_, J2_MIN, J3_MIN,
    					  J1_MIN_, J2_MIN, J3_MIN,
    					  J1_MIN, J2_MIN, J3_MIN,
    					  J1_MIN, J2_MIN, J3_MIN])
    	self.joint_pos_min_B = np.array([ J1_MIN, J2_MIN, J3_MIN,
    					   J1_MIN_, J2_MIN, J3_MIN,
    					   J1_MIN_, J2_MIN, J3_MIN,
    					   J1_MIN, J2_MIN, J3_MIN])
    	self.joint_pos_min_L = np.array([ J1_MIN, J2_MIN, J3_MIN,
    					   J1_MIN, J2_MIN, J3_MIN,
    					   J1_MIN_, J2_MIN, J3_MIN,
    					   J1_MIN_, J2_MIN, J3_MIN])
    	
    	self.starting_pos = np.array([2048,2560,512,
    				       2048,2560,512,
    				       2048,2560,512,
    				       2048,2560,512])
    	
    	self.joint_state = np.zeros(action_dim-2)
    	self.action = np.zeros(action_dim)
    	self.action_x = 0
    	self.action_y = 0
    	self.joint_action = np.zeros(action_dim-2)
    	self.joint_action_pub = np.zeros(action_dim-2)
    	self.stance_joint = np.array([2048,2560,512,
    				       2048,2560,512,
    				       2048,2560,512])
    	self.joint_pos = self.starting_pos
    	#self.joint_state_subscriber = rospy.Subscriber('/legged_robot/joint_states',JointState, self.getJointStates)
    	#for real robot the topic is not attach with legged robot
    	self.joint_state_subscriber = rospy.Subscriber('/joint_states',JointState, self.getJointStates)
    	self.sub_orient = rospy.Subscriber('/imu/data',Imu, self.getOrient)
    	self.sub_ang_vel = rospy.Subscriber('/imu/data',Imu, self.getAngVel)
    	self.sub_acc = rospy.Subscriber('/imu/data',Imu, self.getAcceleration)
    	self.scan = rospy.Subscriber('/scan', LaserScan, self.getScan)
    	
    	self.past_distance = 0.
    	self.stopped = 0
    	#self.force_stop = 0
    	self.tilt_over_stop = 0
    	self.action_dim = action_dim
    	self.action_coeff = 1
    	self.z_limit = 0.042
    	self.ep = 0
    	self.agent_freez = 0
    	self.dir_mode = 0
    	self.index = 0
    	self.index_ = 0 #last index
    	self.acc_x_p = 0
    	self.acc_y_p = 0
    	self.acc_z_p = 0
    	self.dir_error = False
    	self.collide = False
    	self.done_path = False
    	self.pose_record = []
    	self.goal_record = []
    	self.action_record = []
    	self.score = []
    	#self.joint
    	#Keys CTRL + c will stop script
    	rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        #you can stop turtlebot by publishing an empty Twist
        #message
        rospy.loginfo("Stopping Robot")
        self.joint_command_client(self.starting_pos)
        rospy.sleep(1)

    def getGoalDistance(self,x,y,goalx,goaly):
        goal_distance = round(math.hypot(float(goalx) - float(x), float(goaly) - float(y)), 2)
        self.past_distance = goal_distance

        return goal_distance

    # ROBOT POSE NO LONGER RETRIVED FROM SENSOR DIRECTLY, SHOULD CALCULATE 
    # WITH BOTH LINEAR ACCELERATION AND ORIENTATION RESULT FROM IMU!!!
    def getPose(self): 
        roll,pitch,yaw = euler_from_quaternion([self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w])
        acc_x= self.acc_x - self.acc_x_p
        acc_y= self.acc_y - self.acc_y_p
        acc_z= self.acc_z - self.acc_z_p
        w_acx = acc_z*sin(pitch) + acc_x*cos(pitch)*cos(yaw) - acc_y*cos(pitch)*sin(yaw)
        w_acy = acc_x*(cos(roll)*sin(yaw) + cos(yaw)*sin(pitch)*sin(roll)) + acc_y*(cos(roll)*cos(yaw) - sin(pitch)*sin(roll)*sin(yaw)) - acc_z*cos(pitch)*sin(roll)
        w_acz = acc_x*(sin(roll)*sin(yaw) - cos(roll)*cos(yaw)*sin(pitch)) + acc_y*(cos(yaw)*sin(roll) + cos(roll)*sin(pitch)*sin(yaw)) + acc_z*cos(pitch)*cos(roll)
        self.delta_time = rospy.Time.now().secs - self.init_time.secs
        print(self.delta_time)
        self.pose_x += float(w_acx)*((float(self.delta_time))**2)
        self.pose_y += float(w_acy)*((float(self.delta_time))**2)
        self.pose_z = float(w_acz)*((float(self.delta_time))**2)
        self.init_time = rospy.Time.now()
        #self.last_recieved_stamp = rospy.Time.now()
        self.acc_x_p = self.acc_x
        self.acc_y_p = self.acc_y
        self.acc_z_p = self.acc_z
        return self.pose_x , self.pose_y, self.pose_z
        
    def getScan(self,scan):
    	self.scan_data = np.array([])
    	self.scan = scan.intensities
    	for i in range(len(self.scan)):
    		j = i%36
    		if j == 0:
    			self.scan_data = np.append(self.scan_data,self.scan[i])
    			
    	return self.scan_data
    
    
    def getJointStates(self, joint_state):
        self.joint_state = np.array(joint_state.position)
        #self.last_recieved_stamp = rospy.Time.now()

    def save_pose_record(self,pose_):
    	self.pose_record = np.append(self.pose_record,pose_)
    	self.pose_record = self.pose_record.reshape(int(len(self.pose_record)/3),3)
    	
    def get_pose_record(self):
    	pose_record = self.pose_record
    	return pose_record
    	
    def save_goal_record(self,goal_):
    	self.goal_record = np.append(self.goal_record,goal_)
    	self.goal_record = self.goal_record.reshape(int(len(self.goal_record)/2),2)
    	
    def get_goal_record(self):
    	goal_record = self.goal_record
    	return goal_record
    	
    def save_action_record(self,action_js):
    	self.action_record = np.append(self.action_record,action_js)
    	self.action_record = self.action_record.reshape(int(len(self.action_record)/12),12)
    	
    def get_action_js(self):
    	action_record = self.action_record
    	return action_record
    
    def getAcceleration(self, data):
        self.acc_x = data.linear_acceleration.x
        self.acc_y = data.linear_acceleration.y
        self.acc_z = data.linear_acceleration.z
        #self.last_recieved_stamp = rospy.Time.now()

    def getAngVel(self, data):
        self.ang_v_x = data.angular_velocity.x
        self.ang_v_y = data.angular_velocity.y
        self.ang_v_z = data.angular_velocity.z
        #self.last_recieved_stamp = rospy.Time.now()
        
    def getOrient(self, data):
        self.orientation = data.orientation
        self.orientation.x = round(self.orientation.x, 4)
        self.orientation.y = round(self.orientation.y, 4)
        self.orientation.z = round(self.orientation.z, 4)
        self.orientation.w = round(self.orientation.w, 4)
        #self.twist = data.twist[1]
        #self.last_recieved_stamp = rospy.Time.now()

        
    def joint_command_client(self,val):
    	#rospy.sleep(rate_value)
    	#print(str(input_val))
    	self.input_val.goal_position_L1_1 = float(val[0])
    	self.input_val.goal_position_L1_2 = float(val[1])
    	self.input_val.goal_position_L1_3 = float(val[2])
    	self.input_val.goal_position_L2_1 = float(val[3])
    	self.input_val.goal_position_L2_2 = float(val[4])
    	self.input_val.goal_position_L2_3 = float(val[5])
    	self.input_val.goal_position_L3_1 = float(val[6])
    	self.input_val.goal_position_L3_2 = float(val[7])
    	self.input_val.goal_position_L3_3 = float(val[8])
    	self.input_val.goal_position_L4_1 = float(val[9])
    	self.input_val.goal_position_L4_2 = float(val[10])
    	self.input_val.goal_position_L4_3 = float(val[11])
    	print(str(self.input_val))
    	self.joint_pub.call(self.input_val)
    	
    
    def getState(self, scan):
    	#===============get scan data================
        scan_range = []
        for i in range(scan.shape[0]):
            if scan[i] == float('Inf') or scan[i] == float('inf'):
                scan_range.append(3.5)
            elif np.isnan(scan[i]) or scan[i] == float('nan'):
                scan_range.append(0)
            else:
                scan_range.append(scan[i])

        min_range = 0.136
        done = False
        rospy.sleep(1/60) 
        
        if min_range > min(scan_range) > 0:
            #done = True
            print('scan_range_limit_too_low!')
        #===============get direction toward goal================
        self.getPose()
        print('current position x: ',self.pose_x)
        print('current position y: ',self.pose_y)
        print('goal position x: ',self.goal_x)
        print('goal position y: ',self.goal_y)
        
        goal_angle = math.atan2(float(self.goal_y) - float(self.pose_y), float(self.goal_x) - float(self.pose_x))
        roll,pitch,yaw = euler_from_quaternion([self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w])
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        print('yaw: ',self.yaw)
        heading = goal_angle - (self.yaw -(pi/4)) #original robot yaw is 0.785 which is 45 from the right to right top direction (front of robot facing right middle) this orientaton condition is while the robot facing toward front(so another pi/4 is added
        
       
        
        # direction defining
        if heading <=pi/4 and heading >=-pi/4:
        	dir_mode = 6 #walking toward right
        elif heading <= 0.75*pi and heading >= pi/4:
        	dir_mode = 8 #walking toward front
        elif heading <=-pi/4 and heading >= -0.75*pi:
        	dir_mode = 2  # walking toward back
        else:
        	dir_mode = 4 # walking toward left
        
        
        self.dir_mode = dir_mode
        print('state_dir_: ', self.dir_mode)
        rospy.sleep(0.1)
        #===========get the distance needed to travel for reaching the goal================
        current_distance = np.abs(round(math.hypot(float(self.goal_x) - float(self.pose_x), float(self.goal_y) - float(self.pose_y)),2))
        #=========== get the robot base info ================
        pos = [self.pose_x,self.pose_y,self.pose_z]
        ang_vel = [self.ang_v_x, self.ang_v_y, self.ang_v_z]
        l_acc = [self.acc_x,self.acc_y,self.acc_z]
        #rpy = [self.roll, self.pitch, self.yaw]
        rp = [self.roll, self.pitch]
        #===========get the leg joint angle ================
        joint = self.joint_state
        js = []
        for j in range(len(joint)):
        	js.append(joint[j])
           	
        if current_distance < 0.12:
            self.touch_goal = True
            done = True
            
            
        
        return scan_range + ang_vel + l_acc + rp + js + pos + [current_distance] + [dir_mode], done # state size is 34 if without heading

    def setReward(self, state, done):
    	acc_reward = 0
    	orient_reward = 0
    	joint_reward = 0
    	height_reward = 0
    	dis_reward = 0
    	dir_reward = 0
    	#===========set the reward base on orientation preventing collapse================
    	angvel_x = state[10] #angular vel_x from state
    	if np.abs(angvel_x) > 0.390:
    		print('angvel_x: ', angvel_x)
    	
    	angvel_y = state[11] #angular vel_y from state
    	if np.abs(angvel_y) > 0.390:
    		print('angvel_y: ', angvel_y)
    	
    	angvel_z = state[12] #angular vel_z from state
    	if np.abs(angvel_z) > 0.390:
    		print('angvel_z: ', angvel_z)
    		#done = True
    	
    	#angvel_reward = angvelx_r + angvely_r + angvelz_r
    	#===========set the reward base on orientation preventing collapse================
    	acc_x = state[13] #linear acc_x from state
    	print('acc_x: ',acc_x)
    	if  np.abs(acc_x) > 0.390:
    		acc_reward += 0
    		
    	acc_y = state[11] #linear acc_y from state
    	print('acc_y: ',acc_y)
    	if  np.abs(acc_y) > 0.390:
    		acc_reward += 0
    	
    	acc_z = state[12] #linear acc_z from state
    	print('acc_z: ',acc_z)
    	if  np.abs(acc_z) > 0.390:
    		acc_reward += 0
    	
    	#===========set the reward base on orientation preventing collapse================
    	roll_state = state[16] #roll from state
    	pitch_state = state[17] #pitch from state
    	print('roll: ', roll_state)
    	print('pitch: ', pitch_state)
    	
    	if np.abs(roll_state)> 3.0:
    		orient_reward += 0
    	
    	if np.abs(pitch_state) >= 3.0:
    		orient_reward += 0
    	
    	if np.abs(roll_state) >= 3.0 and np.abs(pitch_state) >= 3.0:
    		#done = True
    		print('roll: ', roll_state)
    		print('pitch: ', pitch_state)
    	
    	#===========set the reward base on leg joint state to reduce joint error================
    	j_s = state[18:30]
    	j_a = []
    	j_action = self.new_js
    	
    	for k in range(len(j_action)):
    		j_a.append(j_action[k]) #joint from action
    	
    	joint_rate = (abs(np.subtract(j_a,j_s))).sum()
    	if 1 <= joint_rate:
    		joint_reward = 0
    	else:
    		joint_reward = 50
    	
    	
    		
    	#===set the reward base on distance rate to encourage robot move near to target as the iteration go=======
    	walk_dir = state[-1]
    	current_distance = state[-2]
    	rospy.sleep(0.1)
    	if self.dir_action == walk_dir:
    		dir_reward = 500
    		print('action_dir'+str(self.dir_action))
    		print('actual_dir'+str(walk_dir))
    		print('current_distance ' + str(current_distance))
    		if current_distance < 0.10:
    			self.touch_goal = True
    			done = True
    			
    	else:
    		print('direction error!')
    		self.dir_error = True
    		if current_distance < 0.10:
    			self.touch_goal = True
    			done = True
    			
    		else:
    			done = True
    		
    	pos_z = state[-3]
    	distance_rate = (self.past_distance - current_distance)
    	
    	if distance_rate > 0:
    		print('distance_rate: ', distance_rate)
    		dis_reward = 50.
    		
    	else: #distance_rate < 0:
    		print('distance_rate: ', distance_rate)
    		dis_reward = 0.
    	
    	if current_distance == 0:
    		current_distance = 0.0001
    	
    	if pos_z <= self.z_limit:
    		height_reward = 0
    	else:
    		height_reward = 0
    	
    	dis_reward += 100/current_distance
    	#===set the reward base on heading angle of for encourage robot direct near to target as the iteration go=======
    	#ang_reward = -50*abs(heading)
    	reward = 0.1*joint_reward + 0.50*dis_reward + 0.5*dir_reward + 0.1*height_reward + 0.1*orient_reward + 0.1*acc_reward  #+0.25*angvel_reward
    	self.past_distance = current_distance
    	
    	
    	#a, b, c, d = float('{0:.5f}'.format(self.pose_x)), float('{0:.5f}'.format(self.past_pose.x)), float('{0:.5f}'.format(self.pose_y)), float('{0:.5f}'.format(self.past_pose.y))
    	#if a == b and c == d:
    	#	self.stopped += 1
    	#	if self.stopped == 50: # or self.tilt_over_stop == 2:
    	#		rospy.loginfo('Robot is either in the same 50 times in a row! rerun!')
    	#		self.stopped = 0
    	#		self.force_stop =1
    	#		done = True
    	#else:
    	#	self.stopped = 0
    	
    	
    	# check for the done condition, whether to reset once done or how, if dir correct then how if wrong then how, should reset new goal or update reward and recalculate for reaching the same goal?
    	if done:
    		#rospy.loginfo("Collision!!")
    		if self.dir_error:
    			reward = -25
    			self.score.append(reward)
    			pose_ = self.get_pose_record()
    			joint_ = self.get_action_js()
    			goal_ = self.get_goal_record()
    			file_name1 = 'real_robot_ddpg_pinn_ks_goal_reward_'+ str(self.index) +'.csv'
    			np.savetxt(file_name1, self.score, delimiter=",") # Rewards
    			file_name2 = 'real_robot_ddpg_pinn_ks_pose_'+ str(self.index) +'.csv'
    			np.savetxt(file_name2, pose_, delimiter=",") # Rewards
    			file_name3 = 'real_robot_ddpg_pinn_ks_action_'+ str(self.index) +'.csv'
    			np.savetxt(file_name3, joint_, delimiter=",") # Rewards
    			file_name4 = 'real_robot_ddpg_pinn_ks_goal_pose_'+ str(self.index) +'.csv'
    			np.savetxt(file_name4, goal_, delimiter=",") # Rewards
    			self.index += 1
    			
    		elif self.touch_goal:
    			rospy.loginfo("Goal!!")
    			reward = 1000.
    			self.score.append(reward)
    			pose_ = self.get_pose_record()
    			joint_ = self.get_action_js()
    			goal_ = self.get_goal_record()
    			file_name1 = 'real_robot_ddpg_pinn_ks_goal_reward_'+ str(self.index) +'.csv'
    			np.savetxt(file_name1, self.score, delimiter=",") # Rewards
    			file_name2 = 'real_robot_ddpg_pinn_ks_pose_'+ str(self.index) +'.csv'
    			np.savetxt(file_name2, pose_, delimiter=",") # Rewards
    			file_name3 = 'real_robot_ddpg_pinn_ks_action_'+ str(self.index) +'.csv'
    			np.savetxt(file_name3, joint_, delimiter=",") # Rewards
    			file_name4 = 'real_robot_ddpg_pinn_ks_goal_pose_'+ str(self.index) +'.csv'
    			np.savetxt(file_name4, goal_, delimiter=",") # Rewards
    			self.index += 1
    		else:
    			reward = reward
    	else:
    		reward = reward
    		self.score.append(reward)
    		pose_ = self.get_pose_record()
    		joint_ = self.get_action_js()
    		goal_ = self.get_goal_record()
    		file_name1 = 'real_robot_ddpg_pinn_ks_goal_reward_'+ str(self.index) +'.csv'
    		np.savetxt(file_name1, self.score, delimiter=",") # Rewards
    		file_name2 = 'real_robot_ddpg_pinn_ks_pose_'+ str(self.index) +'.csv'
    		np.savetxt(file_name2, pose_, delimiter=",") # Rewards
    		file_name3 = 'real_robot_ddpg_pinn_ks_action_'+ str(self.index) +'.csv'
    		np.savetxt(file_name3, joint_, delimiter=",") # Rewards
    		file_name4 = 'real_robot_ddpg_pinn_ks_goal_pose_'+ str(self.index) +'.csv'
    		np.savetxt(file_name4, goal_, delimiter=",") # Rewards
    		self.index += 1
    		
    	
    	self.joint_command_client(self.starting_pos)
    	
    	
    	return reward, done
    
    def step(self, action, past_action):
    	print('action:',action)
    	print('before action joint state:',self.joint_state)
    	self.getPose()
    	self.action_x = np.clip(action[0],a_min=float(self.pose_x)-0.05,a_max=float(self.pose_x)+0.05)
    	self.action_y = np.clip(action[1],a_min=float(self.pose_y)-0.05,a_max=float(self.pose_y)+0.05)
    	goal_angle = math.atan2(float(self.action_y) - float(self.pose_y), float(self.action_x) - float(self.pose_x))
    	
    	heading = goal_angle - (self.yaw -(pi/4))
    	
    	print('action_x: ', self.action_x)
    	print('action_y: ', self.action_y)
    	print('action heading angle: ', heading)
    	rospy.sleep(0.06)
    	
    	# direction defining
    	if heading <=pi/4 and heading >=-pi/4:
    		dir_mode = 6 #walking toward right
    		self.joint_pos_min = self.joint_pos_min_R
    		self.joint_pos_max = self.joint_pos_max_R
    	elif heading <= 0.75*pi and heading >= pi/4:
    		dir_mode = 8 #walking toward front
    		self.joint_pos_min = self.joint_pos_min_F
    		self.joint_pos_max = self.joint_pos_max_F
    	elif heading <=-pi/4 and heading >= -0.75*pi:
    		dir_mode = 2  # walking toward back
    		self.joint_pos_min = self.joint_pos_min_B
    		self.joint_pos_max = self.joint_pos_max_B
    	else:
    		dir_mode = 4 # walking toward left
    		self.joint_pos_min = self.joint_pos_min_L
    		self.joint_pos_max = self.joint_pos_max_L
    	
    	self.dir_action = dir_mode
    	print('action_dir_ : ', self.dir_action)
    	rospy.sleep(0.1)
    	self.joint_action = np.clip(action[2:],a_min=self.joint_pos_min,a_max=self.joint_pos_max)
    	print('joint action: ', self.joint_action)
    	w_t = 20
    	self.pose_record = []
    	self.pose_record_all = []
    	self.goal_record = []
    	self.action_record = [] 
    	for t in range(w_t):
    		L11,L12,L13,L21,L22,L23,L31,L32,L33,L41,L42,L43 = self.joint_action
    		if dir_mode == 8: #front
    			if t <= 5:
    				Q11 = L11*CONV_*0.5*(1-cos(12*pi*(t/30))) + 2048
    				Q12 = L12*CONV_*sin(12*pi*(t/30)) + 2560
    				Q13 = L13*CONV_*sin(12*pi*(t/30)) + 512
    				Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    				
    			elif 5< t <=10:
    				Q31 = L31*CONV_*0.5*(1-cos(12*pi*((t-5)/30))) + 2048
    				Q32 = L32*CONV_*sin(12*pi*((t-5)/30)) + 2560
    				Q33 = L33*CONV_*sin(12*pi*((t-5)/30)) + 512
    				Q21,Q22,Q23,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			elif 10< t <=15:
    				Q21 = L21*CONV_*0.5*(1-cos(12*pi*((t-10)/30))) + 2048
    				Q22 = L22*CONV_*sin(12*pi*((t-10)/30)) + 2560
    				Q23 = L23*CONV_*sin(12*pi*((t-10)/30)) + 512
    				Q31,Q32,Q33,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			else:
    				Q41 = L41*CONV_*0.5*(1-cos(12*pi*((t-15)/30))) + 2048
    				Q42 = L42*CONV_*sin(12*pi*((t-15)/30)) + 2560
    				Q43 = L43*CONV_*sin(12*pi*((t-15)/30)) + 512
    				Q21,Q22,Q23,Q11,Q12,Q13,Q31,Q32,Q33 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    				
    		elif dir_mode == 6: #right
    			if t <= 5:
    				Q21 = L21*CONV_*0.5*(1-cos(12*pi*((t)/30))) + 2048
    				Q22 = L22*CONV_*sin(12*pi*((t)/30)) + 2560
    				Q23 = L23*CONV_*sin(12*pi*((t)/30)) + 512
    				Q11,Q12,Q13,Q31,Q32,Q33,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			elif 5< t <=10:
    				Q41 = L41*CONV_*0.5*(1-cos(12*pi*((t-5)/30))) + 2048
    				Q42 = L42*CONV_*sin(12*pi*((t-5)/30)) + 2560
    				Q43 = L43*CONV_*sin(12*pi*((t-5)/30)) + 512
    				Q21,Q22,Q23,Q11,Q12,Q13,Q31,Q32,Q33 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			elif 10< t <=15:
    				Q31 = L31*CONV_*0.5*(1-cos(12*pi*((t-10)/30))) + 2048
    				Q32 = L32*CONV_*sin(12*pi*((t-10)/30)) + 2560
    				Q33 = L33*CONV_*sin(12*pi*((t-10)/30)) + 512
    				Q21,Q22,Q23,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			else:
    				Q11 = L11*CONV_*0.5*(1-cos(12*pi*((t-15)/30))) + 2048
    				Q12 = L12*CONV_*sin(12*pi*((t-15)/30)) + 2560
    				Q13 = L13*CONV_*sin(12*pi*((t-15)/30)) + 512
    				Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    		
    		elif dir_mode == 2: #back
    			if t <= 5:
    				Q31 = L31*CONV_*0.5*(1-cos(12*pi*((t)/30))) + 2048
    				Q32 = L32*CONV_*sin(12*pi*((t)/30)) + 2560
    				Q33 = L33*CONV_*sin(12*pi*((t)/30)) + 512
    				Q21,Q22,Q23,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			elif 5< t <=10:
    				Q11 = L11*CONV_*0.5*(1-cos(12*pi*((t-5)/30))) + 2048
    				Q12 = L12*CONV_*sin(12*pi*((t-5)/30)) + 2560
    				Q13 = L13*CONV_*sin(12*pi*((t-5)/30)) + 512
    				Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			elif 10< t <=15:
    				Q41 = L41*CONV_*0.5*(1-cos(12*pi*((t-10)/30))) + 2048
    				Q42 = L42*CONV_*sin(12*pi*((t-10)/30)) + 2560
    				Q43 = L43*CONV_*sin(12*pi*((t-10)/30)) + 512
    				Q21,Q22,Q23,Q11,Q12,Q13,Q31,Q32,Q33 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			else:
    				Q21 = L21*CONV_*0.5*(1-cos(12*pi*((t-15)/30))) + 2048
    				Q22 = L22*CONV_*sin(12*pi*((t-15)/30)) + 2560
    				Q23 = L23*CONV_*sin(12*pi*((t-15)/30)) + 512
    				Q31,Q32,Q33,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    				
    		else: #left
    			if t <= 5:
    				Q41 = L41*CONV_*0.5*(1-cos(12*pi*((t)/30))) + 2048
    				Q42 = L42*CONV_*sin(12*pi*((t)/30)) + 2560
    				Q43 = L43*CONV_*sin(12*pi*((t)/30)) + 512
    				Q21,Q22,Q23,Q11,Q12,Q13,Q31,Q32,Q33 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			elif 5< t <=10:
    				Q21 = L21*CONV_*0.5*(1-cos(12*pi*((t-5)/30))) + 2048
    				Q22 = L22*CONV_*sin(12*pi*((t-5)/30)) + 2560
    				Q23 = L23*CONV_*sin(12*pi*((t-5)/30)) + 512
    				Q31,Q32,Q33,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			elif 10< t <=15:
    				Q11 = L11*CONV_*0.5*(1-cos(12*pi*((t-10)/30))) + 2048
    				Q12 = L12*CONV_*sin(12*pi*((t-10)/30)) + 2560
    				Q13 = L13*CONV_*sin(12*pi*((t-10)/30)) + 512
    				Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			else:
    				Q31 = L31*CONV_*0.5*(1-cos(12*pi*((t-15)/30))) + 2048
    				Q32 = L32*CONV_*sin(12*pi*((t-15)/30)) + 2560
    				Q33 = L33*CONV_*sin(12*pi*((t-15)/30)) + 512
    				Q21,Q22,Q23,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.joint_command_client(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.001)
    				self.pose_x,self.pose_y,self.pose_z = self.getPose()
    				self.save_pose_record([self.pose_x,self.pose_y,self.pose_z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    				
    		
    	
    	
    	print('after action joint state:',self.joint_state)
    	self.new_js = self.joint_action_pub
    	rospy.sleep(0.5)
    	data = None
    	
    	while data is None:
    		try:
    			data = self.scan_data
    		except:
    			pass
    	
    	pose_ = self.get_pose_record()
    	joint_ = self.get_action_js()
    	
    	pose_file = 'real_robot_ddpg_pinn_ks_pose_for_dir_'+ str(self.dir_action) +'.csv'
    	np.savetxt(pose_file, pose_, delimiter=",") # pose for directional ,2(backward),4(left),6(right),8(forward)
    	joint_pose_ = 'real_robot_ddpg_pinn_ks_joint_pose_for_dir_'+ str(self.dir_action) +'.csv'
    	np.savetxt(joint_pose_, joint_, delimiter=",") # action joint for directional ,2(backward),4(left),6(right),8(forward)
    	
    	state, done = self.getState(data)
    	rospy.sleep(0.5)
    	reward, done = self.setReward(state, done)
    	
    	return np.asarray(state), reward, done

    #This is real robot experiment, there is no need to reset, everytime requesting for reset should be done for one espisode,
    #which either the robot reach the target or the step number is hit to the maximum limit, reward and penalty were given before enter this module 
    def reset(self):
    	self.init_time = rospy.Time.now()
    	self.joint_pos = self.starting_pos
    	self.joint_command_client(self.starting_pos)
    	dx = uniform(-0.5,0.5)
    	dy = uniform(-0.5,0.5)
    	self.pose_x,self.pose_y,self.pose_z = self.getPose()
    	goal_pose = Pose()
    	goal_pose.position.x = float(self.pose_x) + float(dx)
    	goal_pose.position.y = float(self.pose_y) + float(dy)
    	goal_pose.position.z = float(self.pose_z)
    	self.goal_x = goal_pose.position.x
    	self.goal_y = goal_pose.position.y
    	
    	data = self.scan_data
    	print(data)
    	
    	#while data is None:
    	#	try:
    	#		# subscribed scan data on top
    	#		#data = rospy.wait_for_message('scan', LaserScan, timeout=5) #the data here will become the scan data and the imu data
    	#		data = self.getScan(self.scan)
    	#		print(data)
    	#	except:
    	#		pass
    	
    	rospy.sleep(0.1)
    	self.pose_x,self.pose_y,self.pose_z = self.getPose()
    	self.goal_distance = self.getGoalDistance(self.pose_x,self.pose_y,self.goal_x,self.goal_y)
    	state, _ = self.getState(data)
    	
    	return np.asarray(state)
