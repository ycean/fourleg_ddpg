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

import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('fourleg_ddpg/src',
                                                'turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        #self.stage = rospy.get_param('/stage_number')
        self.goal_position = Pose()
        self.init_goal_x = 0.1551
        self.init_goal_y = -0.0366
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 0.6
        self.obstacle_4 = -0.6, -0.6
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.index = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self,x,y):
    	self.goal_position.position.x = x
    	self.goal_position.position.y = y
    	while True:
    		if not self.check_model:
    			rospy.wait_for_service('gazebo/spawn_sdf_model')
    			spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
    			spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
    			rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x, self.goal_position.position.y)
    			break
    		else:
    			pass

    def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass
                
    def getIndex(self):
    	return self.index

    def getPosition(self,index=0, goal=False, position_check=False, delete=False):
        print("goal is: ", goal)
        if delete:
        	self.deleteModel()
        	print("done delete goal model")
        while position_check:
        	
        	#TEST FOR DDPG (ddpg_36 got selected)
        	#goal_x_list = [0.0355,0.1312,0.1874,0.2525,0.188,0.1898,0.1962,0.0529,-0.0932,-0.2248,-0.2459,-0.3404,-0.2484]
        	#goal_y_list = [0.1750,0.27,0.4127,0.5637,0.6588,0.8364,1.0047,1.059,1.11866,1.17106,1.01756,1.0532,0.9348]
        	#TEST FOR DDPG+PINN_BURGER (PINN_N_CHANGE_14 got selected)
        	#goal_x_list = [0.1741,0.3242,0.4161,0.5324,0.6444,0.7449,0.8502,0.9622,1.0743,0.9515,1.0589,0.9713,1.0456,0.9273,1.0206,0.9209]
        	#goal_y_list = [-0.0574,-0.1771,-0.1057,-0.0144,0.0593,0.1498,0.2417,0.3553,0.452,0.5834,0.6837,0.8082,0.9054,0.969,1.0662,0.9828]
        	#TEST FOR DDPG+PINN_KS (PINN_KS_CHANGE_3 got selected)
        	#goal_x_list = [-0.1652,-0.2970,-0.3689,-0.5467,-0.7058,-0.8549,-1.0052,-1.0285,-1.0526,-1.0759,-0.9967,-1.0208,-0.8632,1]
        	#goal_y_list = [-0.0352,-0.0813,-0.2283,-0.2445,-0.2799,-0.3121,-0.3476,-0.4971,-0.6499,-0.797,-0.8586,-1.0103,-1.0201,1]
        	#TEST FOR DDPG_PINN_KDV
        	goal_x_list = [0.1551,0.3005,0.4232,0.528,0.6227,0.7286,0.8389,0.9895,1.0493,0.9346,1.0348,0.9339,1.0512,0.9522,0]
        	goal_y_list = [-0.0366,-0.0832,0.033,0.1528,0.2859,0.4072,0.5275,0.4961,0.6591,0.7186,0.8402,0.9219,1.0498,0.9271,0]
        	
        	if goal:
        		self.index = index
        		position_check = False
        	elif not goal:
        		if self.last_index >= 16:
        			self.index = 0
        			self.last_index = 0
        			position_check = False
        		else:
        			self.index = index
        			position_check = False
        	
        	print(self.index, self.last_index)
        	self.goal_position.position.x = goal_x_list[self.index]
        	self.goal_position.position.y = goal_y_list[self.index]
        		

        time.sleep(0.5)
        self.last_index = self.index
        self.respawnModel(self.goal_position.position.x,self.goal_position.position.y)

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y
