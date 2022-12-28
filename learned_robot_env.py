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
from std_msgs.msg import String
from std_msgs.msg import Float64
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryActionGoal, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Twist, Transform, TransformStamped, Vector3
from sensor_msgs.msg import Imu, LaserScan, JointState
from gazebo_msgs.srv import SetModelState, DeleteModel
from gazebo_msgs.msg import LinkStates, ContactState, ModelState
from math import sin, cos, pi, pow, atan2, sqrt, ceil, atan, hypot
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
world = False
if world:
    from respawnGoal import Respawn
else:
    from respawnGoal import Respawn
import copy
target_not_movable = False

J1_MAX_ = -0.785
J1_MIN_ = -1.2
J1_MAX = 1.2 #radian
J2_MAX = 0.2094
J3_MAX = -0.1675
J1_MIN = 0.785 #radian
J2_MIN = 0.1675
J3_MIN = -0.2094

class LegJoints:
    def __init__(self,joint_ls):
        self.jac = actionlib.SimpleActionClient('/legged_robot/joint_trajectory_controller/follow_joint_trajectory',
                                                FollowJointTrajectoryAction)
        rospy.loginfo('Waiting for joint trajectory action')
        self.jac.wait_for_server()
        rospy.loginfo('Found joint trajectory action!')
        self.jpub = rospy.Publisher('/legged_robot/joint_trajectory_controller/command',JointTrajectory,queue_size=1)
        self.joint_ls = joint_ls
        self.jpub_zeros = np.zeros(len(joint_ls))
        self.jpub_vel = 0.75 * np.ones(len(joint_ls))
        self.jpub_eff = 1.50 * np.ones(len(joint_ls))

    def move(self, pos):
        msg = FollowJointTrajectoryActionGoal()
        msg.goal.trajectory.joint_names = self.joint_ls
        point = JointTrajectoryPoint()
        point.positions = pos
        point.time_from_start = rospy.Duration(0.08)
        msg.goal.trajectory.points.append(point)
        self.jac.send_goal_and_wait(msg.goal)

    def move_jtp(self, pos):
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_ls
        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = self.jpub_vel
        point.accelerations = self.jpub_zeros
        point.effort = self.jpub_eff
        point.time_from_start = rospy.Duration(0.08)
        jtp_msg.points.append(point)
        self.jpub.publish(jtp_msg)

    def reset_move_jtp(self, pos):
        jtp_msg = JointTrajectory()
        self.jpub.publish(jtp_msg)
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_ls
        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = self.jpub_zeros
        point.accelerations = self.jpub_zeros
        point.effort = self.jpub_zeros
        point.time_from_start = rospy.Duration(0.08)
        jtp_msg.points.append(point)
        self.jpub.publish(jtp_msg)


class Env():
    def __init__(self, action_dim=12):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.initGoal = True
        self.get_goalbox = False
        self.goal = False
        self.position = Pose()
        self.model_state = ModelState()
        self.pose = LinkStates()
        self.joint_state = JointState()
        self.model_state.model_name = 'legged_robot'
        self.model_state.reference_frame = 'world'
        self.model_state.pose.orientation.z = 0.3826834
        self.model_state.pose.orientation.w = 0.9238795
        

        self.joint_ls = ['leg1_j1', 'leg1_j2', 'leg1_j3',
                         'leg2_j1', 'leg2_j2', 'leg2_j3',
                         'leg3_j1', 'leg3_j2', 'leg3_j3',
                         'leg4_j1', 'leg4_j2', 'leg4_j3']
        self.leg_joints = LegJoints(self.joint_ls)
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        #self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.pose_subscriber = rospy.Subscriber('/gazebo/link_states',LinkStates, self.getPose)
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
        
        self.starting_pos = np.array([0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0])
                                     
        self.joint_state = np.zeros(action_dim-2)
        self.action = np.zeros(action_dim)
        self.action_x = 0
        self.action_y = 0
        self.joint_action = np.zeros(action_dim-2)
        self.joint_action_pub = np.zeros(action_dim-2)
        self.stance_joint = np.zeros(9)
        self.joint_pos = self.starting_pos
        self.joint_state_subscriber = rospy.Subscriber('/legged_robot/joint_states',JointState, self.getJointStates)
        self.sub_orient = rospy.Subscriber('/legged_robot/imu/data',Imu, self.getOrient)
        self.sub_ang_vel = rospy.Subscriber('/legged_robot/imu/data',Imu, self.getAngVel)
        self.sub_acc = rospy.Subscriber('/legged_robot/imu/data',Imu, self.getAcceleration)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        self.past_distance = 0.
        self.stopped = 0
        self.force_stop = 0
        self.tilt_over_stop = 0
        self.action_dim = action_dim
        self.action_coeff = 1
        self.z_limit = 0.042
        self.batch = 0
        self.agent_freez = 0
        self.dir_mode = 0
        self.index = 0
        self.index_ = 0 #last index
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
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(0.15)

    def getGoalDistance(self):
        goal_distance = round(math.hypot(self.goal_x - self.pose.x, self.goal_y - self.pose.y), 2)
        self.past_distance = goal_distance

        return goal_distance

    def getPose(self, data):
        self.past_pose = copy.deepcopy(self.pose)
        self.pose = data.pose[1].position
        self.pose.x = round(self.pose.x, 4)
        self.pose.y = round(self.pose.y, 4)
        self.pose.z = round(self.pose.z, 4)
        #self.last_recieved_stamp = rospy.Time.now()
        
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

        
    	
    def getState(self, scan, x, y,r,p,yaw):
    	#===============get scan data================
        scan_range = []
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf') or scan.ranges[i] == float('inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]) or scan.ranges[i] == float('nan'):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        min_range = 0.136
        done = False
        rospy.sleep(1/60) 
        
        if min_range > min(scan_range) > 0:
            #done = True
            print('scan_range_limit_too_low!')
        #===============get direction toward goal================
        print('current position x: ',self.pose.x)
        print('current position y: ',self.pose.y)
        print('goal position x: ',self.goal_x)
        print('goal position y: ',self.goal_y)
        #since to check the directional walk before action is match to the action direction, we should check with the pose before the action for the correct heading dirction of walking, but not the after action pose!
        goal_angle = math.atan2(self.goal_y - y, self.goal_x - x)
        #roll,pitch,yaw = euler_from_quaternion([self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w])
        self.roll = r
        self.pitch = p
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
        rospy.sleep(0.15)
        #===========get the distance needed to travel for reaching the goal================
        current_distance = np.abs(round(math.hypot(self.goal_x - self.pose.x, self.goal_y - self.pose.y),2))
        print('current distance : ', current_distance)
        #=========== get the robot base info ================
        pos = [self.pose.x,self.pose.y,self.pose.z]
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
            self.get_goalbox = True
            self.goal = True
        else:
        	self.get_goalbox = False
        	self.goal = False
            
        
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
    		print('action_dir '+str(self.dir_action))
    		print('actual_dir '+str(walk_dir))
    		print('current_distance ' + str(current_distance))
    		if current_distance < 0.10:
    			self.get_goalbox = True
    			self.goal = True
    	else:
    		print('direction error!')
    		self.dir_error = True
    		if current_distance < 0.10:
    			self.get_goalbox = True
    			self.goal = True
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
    	print(dis_reward)
    	#===set the reward base on heading angle of for encourage robot direct near to target as the iteration go=======
    	#ang_reward = -50*abs(heading)
    	reward = 0.1*joint_reward + 0.50*dis_reward + 0.5*dir_reward + 0.1*height_reward + 0.1*orient_reward + 0.1*acc_reward  #+0.25*angvel_reward
    	self.past_distance = current_distance
    	
    	
    	#a, b, c, d = float('{0:.5f}'.format(self.pose.x)), float('{0:.5f}'.format(self.past_pose.x)), float('{0:.5f}'.format(self.pose.y)), float('{0:.5f}'.format(self.past_pose.y))
    	#if a == b and c == d:
    	#	self.stopped += 1
    	#	if self.stopped == 50: # or self.tilt_over_stop == 2:
    	#		rospy.loginfo('Robot is either in the same 50 times in a row! rerun!')
    	#		self.stopped = 0
    	#		self.force_stop =1
    	#		done = True
    	#else:
    	#	self.stopped = 0
    	
    	if done:
    		#rospy.loginfo("Collision!!")
    		if self.dir_error:
    			reward = -25
    		else:
    			reward = reward
    			self.collide = True
    		
    		
    		#TEST FOR DDPG (ddpg_36 got selected)
    		#goal_x_list = [0.0355,0.1312,0.1874,0.2525,0.188,0.1898,0.1962,0.0529,-0.0932,-0.2248,-0.2459,-0.3404,-0.2484]
    		#goal_y_list = [0.1750,0.27,0.4127,0.5637,0.6588,0.8364,1.0047,1.059,1.11866,1.17106,1.01756,1.0532,0.9348]
    		#TEST FOR DDPG+PINN_BURGER (PINN_N_CHANGE_14 got selected)
    		#goal_x_list = [0.1741,0.3242,0.4161,0.5324,0.6444,0.7449,0.8502,0.9622,1.0743,0.9515,1.0589,0.9713,1.0456,0.9273,1.0206,0.9209]
    		#goal_y_list = [-0.0574,-0.1771,-0.1057,-0.0144,0.0593,0.1498,0.2417,0.3553,0.452,0.5834,0.6837,0.8082,0.9054,0.969,1.0662,0.9828]
    		#TEST FOR DDPG+PINN_KS (PINN_KS_CHANGE_3 got selected)
    		#goal_x_list = [-0.1652,-0.2970,-0.3689,-0.5467,-0.7058,-0.8549,-1.0052,-1.0285,-1.0526,-1.0759,-0.9967,-1.0208,-0.8632,1]
    		#goal_y_list = [-0.0352,-0.0813,-0.2283,-0.2445,-0.2799,-0.3121,-0.3476,-0.4971,-0.6499,-0.797,-0.8586,-1.0103,-1.0201,1]
    		#TEST FOR DDPG+PINN_Kdv
    		goal_x_list = [-0.1652,1]
    		goal_y_list = [-0.0352,1]
    		
    		print('index number: ',self.index)
    		
    		if self.index == 0:
    			com_x = 0
    			com_y = 0
    		
    		else:
    			com_x = goal_x_list[self.index-1]
    			com_y = goal_y_list[self.index-1]
    
    		print('com_x: ', com_x)
    		print('com_y: ', com_y)
    		self.model_state.pose.position.x = com_x
    		self.model_state.pose.position.y = com_y
    		self.model_state.pose.position.z = 0.139
    		
    		self.goal = False
    		self.pub_cmd_vel.publish(Twist())
    		self.reset(self.goal)
    		#self.batch += 1
    		done = False
    	
    	
    	self.index_ = self.index	
    	
    	if self.get_goalbox:
    		
    		rospy.loginfo("Succeed arrive at step "+str(self.index))
    		reward = 1000.
    		self.pub_cmd_vel.publish(Twist())
    		self.score.append(reward)
    		pose_ = self.get_pose_record()
    		joint_ = self.get_action_js()
    		goal_ = self.get_goal_record()
    		file_name1 = 'ddpg_pinn3_goal_reward_'+ str(self.index) +'.csv'
    		np.savetxt(file_name1, self.score, delimiter=",") # Rewards
    		file_name2 = 'ddpg_pinn3_pose_'+ str(self.index) +'.csv'
    		np.savetxt(file_name2, pose_, delimiter=",") # Rewards
    		file_name3 = 'ddpg_pinn3_action_'+ str(self.index) +'.csv'
    		np.savetxt(file_name3, joint_, delimiter=",") # Rewards
    		file_name4 = 'ddpg_pinn3_goal_pose_'+ str(self.index) +'.csv'
    		np.savetxt(file_name4, goal_, delimiter=",") # Rewards
    		
    		goal = self.goal
    		self.index_ = self.index
    		self.index += 1
    		
    		if self.index >= 16:
    		#if self.index > 49:
    			print("pause for completion!")
    			rospy.sleep(5.0)
    			done = True
    			self.done_path = True
    			
    		else:
    			done = False
    			self.goal_x, self.goal_y= self.respawn_goal.getPosition(self.index,goal,True, delete=True)
    			self.goal_distance = self.getGoalDistance()
    			
    		#self.reset()
    		#self.batch += 1
    		print('index no.: ', self.index)
    		self.get_goalbox = False
    	
    	
    	
    	if self.index == self.index_:
    		self.batch += 1
    		self.goal = False
    	else:
    		self.batch = 0
    		
    	if self.batch > 5:
    		done = True
    		self.goal = False
    		self.batch = 0
    	
    	if self.done_path:
    		self.goal = False
    		self.index = 0
    		self.index_ = 0
    		self.batch = 0
    	
    	print('done status: ', done)
    	print('batch number: ', self.batch)
    	print('index number: ', self.index)
    	print('past index number: ', self.index_)
    	
    	return reward, done
    
    def step(self, action, past_action):
    	print('action:',action)
    	print('before action joint state:',self.joint_state)
    	self.action_x = np.clip(action[0],a_min=self.pose.x-0.05,a_max=self.pose.x+0.05)
    	self.action_y = np.clip(action[1],a_min=self.pose.y-0.05,a_max=self.pose.y+0.05)
    	goal_angle = math.atan2(self.action_y - self.pose.y, self.action_x - self.pose.x)
    	robot_com_x = self.pose.x
    	robot_com_y = self.pose.y
    	roll,pitch,yaw = euler_from_quaternion([self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w])
    	heading = goal_angle - (yaw -(pi/4))
    	
    	print('action_x: ', self.action_x)
    	print('action_y: ', self.action_y)
    	print('action heading angle: ', heading)
    	rospy.sleep(0.15)
    	
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
    	w_t = 240
    	self.pose_record = []
    	self.goal_record = []
    	self.action_record = [] 
    	for t in range(w_t):
    		L11,L12,L13,L21,L22,L23,L31,L32,L33,L41,L42,L43 = self.joint_action
    		if dir_mode == 8: #front
    			if t <= 60:
    				Q11 = L11*0.5*(1-cos(pi*(t/30)))
    				Q12 = L12*sin(pi*(t/30))
    				Q13 = L13*sin(pi*(t/30))
    				Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			elif 60< t <=120:
    				Q31 = L31*0.5*(1-cos(pi*((t-60)/30)))
    				Q32 = L32*sin(pi*((t-60)/30))
    				Q33 = L33*sin(pi*((t-60)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			elif 120< t <=180:
    				Q21 = L21*0.5*(1-cos(pi*((t-120)/30)))
    				Q22 = L22*sin(pi*((t-120)/30))
    				Q23 = L23*sin(pi*((t-120)/30))
    				Q31,Q32,Q33,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			else:
    				Q41 = L41*0.5*(1-cos(pi*((t-180)/30)))
    				Q42 = L42*sin(pi*((t-180)/30))
    				Q43 = L43*sin(pi*((t-180)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q31,Q32,Q33 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    				
    		elif dir_mode == 6: #right
    			if t <= 60:
    				Q21 = L21*0.5*(1-cos(pi*((t)/30)))
    				Q22 = L22*sin(pi*((t)/30))
    				Q23 = L23*sin(pi*((t)/30))
    				Q11,Q12,Q13,Q31,Q32,Q33,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			elif 60< t <=120:
    				Q41 = L41*0.5*(1-cos(pi*((t-60)/30)))
    				Q42 = L42*sin(pi*((t-60)/30))
    				Q43 = L43*sin(pi*((t-60)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q31,Q32,Q33 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			elif 120< t <=180:
    				Q31 = L31*0.5*(1-cos(pi*((t-120)/30)))
    				Q32 = L32*sin(pi*((t-120)/30))
    				Q33 = L33*sin(pi*((t-120)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			else:
    				Q11 = L11*0.5*(1-cos(pi*((t-180)/30)))
    				Q12 = L12*sin(pi*((t-180)/30))
    				Q13 = L13*sin(pi*((t-180)/30))
    				Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    		
    		elif dir_mode == 2: #back
    			if t <= 60:
    				Q31 = L31*0.5*(1-cos(pi*((t)/30)))
    				Q32 = L32*sin(pi*((t)/30))
    				Q33 = L33*sin(pi*((t)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			elif 60< t <=120:
    				Q11 = L11*0.5*(1-cos(pi*((t-60)/30)))
    				Q12 = L12*sin(pi*((t-60)/30))
    				Q13 = L13*sin(pi*((t-60)/30))
    				Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			elif 120< t <=180:
    				Q41 = L41*0.5*(1-cos(pi*((t-120)/30)))
    				Q42 = L42*sin(pi*((t-120)/30))
    				Q43 = L43*sin(pi*((t-120)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q31,Q32,Q33 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			else:
    				Q21 = L21*0.5*(1-cos(pi*((t-180)/30)))
    				Q22 = L22*sin(pi*((t-180)/30))
    				Q23 = L23*sin(pi*((t-180)/30))
    				Q31,Q32,Q33,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    				
    		else: #left
    			if t <= 60:
    				Q41 = L41*0.5*(1-cos(pi*((t)/30)))
    				Q42 = L42*sin(pi*((t)/30))
    				Q43 = L43*sin(pi*((t)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q31,Q32,Q33 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			elif 60< t <=120:
    				Q21 = L21*0.5*(1-cos(pi*((t-60)/30)))
    				Q22 = L22*sin(pi*((t-60)/30))
    				Q23 = L23*sin(pi*((t-60)/30))
    				Q31,Q32,Q33,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			elif 120< t <=180:
    				Q11 = L11*0.5*(1-cos(pi*((t-120)/30)))
    				Q12 = L12*sin(pi*((t-120)/30))
    				Q13 = L13*sin(pi*((t-120)/30))
    				Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    			else:
    				Q31 = L31*0.5*(1-cos(pi*((t-180)/30)))
    				Q32 = L32*sin(pi*((t-180)/30))
    				Q33 = L33*sin(pi*((t-180)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				self.save_action_record(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				self.save_pose_record([self.pose.x,self.pose.y,self.pose.z])
    				self.save_goal_record([self.goal_x,self.goal_y])
    				
    		
    	
    	#print('joint pos:',self.joint_action_pub)
    	print('after action joint state:',self.joint_state)
    	self.new_js = self.joint_action_pub
    	rospy.sleep(0.15)
    	data = None
    	
    	while data is None:
    		try:
    			data = rospy.wait_for_message('scan', LaserScan, timeout=5)
    		except:
    			pass
    	
    	
    	pose_ = self.get_pose_record()
    	joint_ = self.get_action_js()
    	
    	pose_file = 'ddpg_pinn3_pose_for_dir_'+ str(self.dir_action) +'.csv'
    	np.savetxt(pose_file, pose_, delimiter=",") # pose for directional ,2(backward),4(left),6(right),8(forward)
    	joint_pose_ = 'ddpg_pinn3_joint_pose_for_dir_'+ str(self.dir_action) +'.csv'
    	np.savetxt(joint_pose_, joint_, delimiter=",") # action joint for directional ,2(backward),4(left),6(right),8(forward)
    	
    	state, done = self.getState(data, robot_com_x,robot_com_y,roll,pitch,yaw)
    	rospy.sleep(0.15)
    	reward, done = self.setReward(state, done)
    	
    	
    	print("finished state class")
    	goal = self.goal
    	return np.asarray(state), reward, done, goal

    def reset(self, goal):
    	#pause physics
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
        except (rospy.ServiceException) as e:
            print('/gazebo/pause_physics service call failed')
            
            
        if self.initGoal:
        	rospy.wait_for_service('gazebo/reset_simulation')
        	try:
        		self.reset_proxy()
        	except (rospy.ServiceException) as e:
        		print("gazebo/reset_simulation service call failed")
        	
        elif self.dir_error or self.collide:
        	print("reset model state!")
        	respawn = True
        	try:
        		rospy.wait_for_service('/gazebo/set_model_state')
        		set_model_state_prox = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        		set_model_state_prox(self.model_state)
        		rospy.loginfo("Robot position : %.1f, %.1f, %.1f", self.model_state.pose.position.x,self.model_state.pose.position.y,self.model_state.pose.position.z)
        		print("done reset model state!")
        	except (rospy.ServiceException) as e:
        		print("gazebo/set_model_state service call failed") 
        	
        else:
        	print('Check goal status:'+ goal)
        	
               
        #self.goal_x, self.goal_y = self.respawn_goal.getPosition(goal)
        self.joint_pos = self.starting_pos
        self.leg_joints.reset_move_jtp(self.starting_pos)
        print("unpause physics!")
        #unpause physics
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print('/gazebo/unpause_physics service call failed')

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(goal)
            self.initGoal = False
        #elif respawn:
        #    
        goal_x = self.goal_x
        goal_y = self.goal_y

        
        print("goal pose :" + str(self.goal_x) + "," + str(self.goal_y))
        rospy.sleep(0.15)
        com_x = self.pose.x
        com_y = self.pose.y
        roll,pitch,yaw = euler_from_quaternion([self.orientation.x,self.orientation.y,self.orientation.z,self.orientation.w])
        state, _ = self.getState(data,com_x,com_y,roll,pitch,yaw)

        return np.asarray(state)
