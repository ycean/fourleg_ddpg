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
        self.init_goal_x = 0.64355
        self.init_goal_y = 0.0
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

    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                              self.goal_position.position.y)
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

    def getPosition(self, ep, position_check=False, delete=False):
        print("ep is: ", ep)
        if delete:
        	self.deleteModel()
        while position_check:
        	goal_x_list1 = [0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355]
        	goal_y_list1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        	goal_x_list2 = [-0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355]
        	goal_y_list2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        	goal_x_list3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        	goal_y_list3 = [0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355, 0.674355]
        	goal_x_list4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        	goal_y_list4 = [-0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355, -0.674355]
        	goal_x_list5 = [0, 0.5, 0.5, -0.5, -0.5, 0, 0.3, 0.874355, 0.433, -0.675, 0.135, -0.628, 0.817]
        	goal_y_list5 = [0.5, 0, 0.25, 0, -0.3, -0.874355, -0.874355, -0.874355, 0.874355, -0.874355, 0.25, 0.5238, -0.2387]
        	
        	self.index = random.randrange(0, 13)
        	print(self.index, self.last_index)
        	if self.last_index == self.index:
        		position_check = True
        	else:
        		self.last_index = self.index
        		position_check = False
        	
        	if 0 <= ep <=199:
        		self.goal_position.position.x = goal_x_list5[self.index]
        		self.goal_position.position.y = goal_y_list5[self.index]
        	elif 200 <= ep <=399:
        		self.goal_position.position.x = goal_x_list5[self.index]
        		self.goal_position.position.y = goal_y_list5[self.index]
        	elif 400 <= ep <=599:
        		self.goal_position.position.x = goal_x_list5[self.index]
        		self.goal_position.position.y = goal_y_list5[self.index]
        	elif 600<= ep <=799:
        		self.goal_position.position.x = goal_x_list5[self.index]
        		self.goal_position.position.y = goal_y_list5[self.index]
        	else:
        		self.goal_position.position.x = goal_x_list5[self.index]
        		self.goal_position.position.y = goal_y_list5[self.index]
        		

        time.sleep(0.5)
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y
