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
from gazebo_msgs.msg import LinkStates, ContactState
from math import sin, cos, pi, pow, atan2, sqrt, ceil, atan, hypot
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
world = False
if world:
    from respawnGoal_ import Respawn
else:
    from respawnGoal_ import Respawn
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
        self.position = Pose()
        self.pose = LinkStates()
        self.joint_state = JointState()

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
        self.ep = 0
        self.agent_freez = 0
        self.dir_mode = 0
        #self.joint
        #Keys CTRL + c will stop script
        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        #you can stop turtlebot by publishing an empty Twist
        #message
        rospy.loginfo("Stopping Robot")
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(1)

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

        
    	
    def getState(self, scan):
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
        
        goal_angle = math.atan2(self.goal_y - self.pose.y, self.goal_x - self.pose.x)
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
        current_distance = np.abs(round(math.hypot(self.goal_x - self.pose.x, self.goal_y - self.pose.y),2))
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
           	
        if current_distance < 0.10:
            self.get_goalbox = True
            
        
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
    	rospy.sleep(0.1)
    	if self.dir_action == walk_dir:
    		dir_reward = 500
    		print('action_dir'+str(self.dir_action))
    		print('actual_dir'+str(walk_dir))
    	else:
    		done = True
    		
    	current_distance = state[-2]
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
    	
    	
    	a, b, c, d = float('{0:.5f}'.format(self.pose.x)), float('{0:.5f}'.format(self.past_pose.x)), float('{0:.5f}'.format(self.pose.y)), float('{0:.5f}'.format(self.past_pose.y))
    	if a == b and c == d:
    		self.stopped += 1
    		if self.stopped == 50: # or self.tilt_over_stop == 2:
    			rospy.loginfo('Robot is either in the same 50 times in a row! rerun!')
    			self.stopped = 0
    			self.force_stop =1
    			done = True
    	else:
    		self.stopped = 0
    	
    	if done:
    		#rospy.loginfo("Collision!!")
    		if self.force_stop ==1:
    			reward = 0
    		else:
    			reward = -25.
    		
    		self.force_stop = 0
    		self.pub_cmd_vel.publish(Twist())
    	
    	if self.get_goalbox:
    		rospy.loginfo("Goal!!")
    		reward = 1000.
    		self.pub_cmd_vel.publish(Twist())
    		
    		if world:
    			ep = self.ep
    			self.goal_x, self.goal_y = self.respawn_goal.getPosition(ep,True, delete=True, running=True)
    			
    			if target_not_movable:
    				self.agent_freez = 1
    				self.reset()
    				
    			else:
    				self.agent_freez = 0
    		else:
    			ep = self.ep
    			self.goal_x, self.goal_y = self.respawn_goal.getPosition(ep,True, delete=True)
    			
    		self.goal_distance = self.getGoalDistance()
    		self.get_goalbox = False
    	
    	return reward, done
    
    def step(self, action, past_action):
    	print('action:',action)
    	print('before action joint state:',self.joint_state)
    	self.action_x = np.clip(action[0],a_min=self.pose.x-0.05,a_max=self.pose.x+0.05)
    	self.action_y = np.clip(action[1],a_min=self.pose.y-0.05,a_max=self.pose.y+0.05)
    	goal_angle = math.atan2(self.action_y - self.pose.y, self.action_x - self.pose.x)
    	
    	heading = goal_angle - (self.yaw -(pi/4))
    	
    	print('action_x: ', self.action_x)
    	print('action_y: ', self.action_y)
    	print('action heading angle: ', heading)
    	rospy.sleep(3.0)
    	
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
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    			elif 60< t <=120:
    				Q31 = L31*0.5*(1-cos(pi*((t-60)/30)))
    				Q32 = L32*sin(pi*((t-60)/30))
    				Q33 = L33*sin(pi*((t-60)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    			elif 120< t <=180:
    				Q21 = L21*0.5*(1-cos(pi*((t-120)/30)))
    				Q22 = L22*sin(pi*((t-120)/30))
    				Q23 = L23*sin(pi*((t-120)/30))
    				Q31,Q32,Q33,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    			else:
    				Q41 = L41*0.5*(1-cos(pi*((t-180)/30)))
    				Q42 = L42*sin(pi*((t-180)/30))
    				Q43 = L43*sin(pi*((t-180)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q31,Q32,Q33 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				
    		elif dir_mode == 6: #right
    			if t <= 60:
    				Q21 = L21*0.5*(1-cos(pi*((t)/30)))
    				Q22 = L22*sin(pi*((t)/30))
    				Q23 = L23*sin(pi*((t)/30))
    				Q11,Q12,Q13,Q31,Q32,Q33,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    			elif 60< t <=120:
    				Q41 = L41*0.5*(1-cos(pi*((t-60)/30)))
    				Q42 = L42*sin(pi*((t-60)/30))
    				Q43 = L43*sin(pi*((t-60)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q31,Q32,Q33 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    			elif 120< t <=180:
    				Q31 = L31*0.5*(1-cos(pi*((t-120)/30)))
    				Q32 = L32*sin(pi*((t-120)/30))
    				Q33 = L33*sin(pi*((t-120)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    			else:
    				Q11 = L11*0.5*(1-cos(pi*((t-180)/30)))
    				Q12 = L12*sin(pi*((t-180)/30))
    				Q13 = L13*sin(pi*((t-180)/30))
    				Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    		
    		elif dir_mode == 2: #back
    			if t <= 60:
    				Q31 = L31*0.5*(1-cos(pi*((t)/30)))
    				Q32 = L32*sin(pi*((t)/30))
    				Q33 = L33*sin(pi*((t)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    			elif 60< t <=120:
    				Q11 = L11*0.5*(1-cos(pi*((t-60)/30)))
    				Q12 = L12*sin(pi*((t-60)/30))
    				Q13 = L13*sin(pi*((t-60)/30))
    				Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    			elif 120< t <=180:
    				Q41 = L41*0.5*(1-cos(pi*((t-120)/30)))
    				Q42 = L42*sin(pi*((t-120)/30))
    				Q43 = L43*sin(pi*((t-120)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q31,Q32,Q33 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    			else:
    				Q21 = L21*0.5*(1-cos(pi*((t-180)/30)))
    				Q22 = L22*sin(pi*((t-180)/30))
    				Q23 = L23*sin(pi*((t-180)/30))
    				Q31,Q32,Q33,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				
    		else: #left
    			if t <= 60:
    				Q41 = L41*0.5*(1-cos(pi*((t)/30)))
    				Q42 = L42*sin(pi*((t)/30))
    				Q43 = L43*sin(pi*((t)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q31,Q32,Q33 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    			elif 60< t <=120:
    				Q21 = L21*0.5*(1-cos(pi*((t-60)/30)))
    				Q22 = L22*sin(pi*((t-60)/30))
    				Q23 = L23*sin(pi*((t-60)/30))
    				Q31,Q32,Q33,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    			elif 120< t <=180:
    				Q11 = L11*0.5*(1-cos(pi*((t-120)/30)))
    				Q12 = L12*sin(pi*((t-120)/30))
    				Q13 = L13*sin(pi*((t-120)/30))
    				Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    			else:
    				Q31 = L31*0.5*(1-cos(pi*((t-180)/30)))
    				Q32 = L32*sin(pi*((t-180)/30))
    				Q33 = L33*sin(pi*((t-180)/30))
    				Q21,Q22,Q23,Q11,Q12,Q13,Q41,Q42,Q43 = self.stance_joint
    				self.joint_action_pub = [Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33,Q41,Q42,Q43]
    				self.leg_joints.move_jtp(self.joint_action_pub)
    				print('joint_moving: ', self.joint_action_pub)
    				rospy.sleep(0.05)
    				
    		
    	
    	#print('joint pos:',self.joint_action_pub)
    	print('after action joint state:',self.joint_state)
    	self.new_js = self.joint_action_pub
    	rospy.sleep(0.5)
    	data = None
    	
    	while data is None:
    		try:
    			data = rospy.wait_for_message('scan', LaserScan, timeout=5)
    		except:
    			pass
    	
    	state, done = self.getState(data)
    	rospy.sleep(0.5)
    	reward, done = self.setReward(state, done)
    	
    	return np.asarray(state), reward, done

    def reset(self):
    	#pause physics
        #rospy.wait_for_service('/gazebo/pause_physics')
        #try:
        #    self.pause_proxy()
        #except (rospy.ServiceException) as e:
        #    print('/gazebo/pause_physics service call failed')
            
        if self.agent_freez == 0:
        	self.ep += 1
        	
        else:
        	self.ep = self.ep
        
        rospy.wait_for_service('gazebo/reset_simulation')
        
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
               
        self.joint_pos = self.starting_pos
        self.leg_joints.reset_move_jtp(self.starting_pos)
        
        #unpause physics
        #rospy.wait_for_service('/gazebo/unpause_physics')
        #try:
        #    self.unpause_proxy()
        #except (rospy.ServiceException) as e:
        #    print('/gazebo/unpause_physics service call failed')

        #rospy.sleep(0.5)
        
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            ep = self.ep
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(ep)
            self.initGoal = False
        else:
            ep = self.ep
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(ep,True, delete=True)

        rospy.sleep(0.9)
        self.goal_distance = self.getGoalDistance()
        state, _ = self.getState(data)

        return np.asarray(state)
