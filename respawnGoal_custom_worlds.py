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
        self.modelPath = self.modelPath.replace('project/src',
                                                'turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        #stage 1 = TCC_world_obst
        #stage 2 = TCC_world_U
        self.stage = rospy.get_param('/stage_number')
        self.goal_position = Pose()
        if self.stage == 1:  
            self.init_goal_x = 0.975166
            self.init_goal_y = -0.790902
        if self.stage == 2:  
            self.init_goal_x = 2.25
            self.init_goal_y = -2.40
        if self.stage == 3:  
            self.init_goal_x = 2.25
            self.init_goal_y = -2.40
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
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

    def getPosition(self, position_check=False, delete=False, running=False):
        if delete:
            self.deleteModel()

        if self.stage == 1:
            while position_check:
                
                goal_x_list = [0.874355, 0.874355, 0.874355, 0.874355, 0.874355, 0.874355, 0.874355, 0.874355, 0.874355, 0.874355, 0.874355]
                goal_y_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                if not running:
                    self.index = 0
                    position_check = False
                    self.goal_position.position.x = goal_x_list[self.index]
                    self.goal_position.position.y = goal_y_list[self.index]
                elif self.index < 10:
                    self.index += 1
                    position_check = False
                    self.goal_position.position.x = goal_x_list[self.index]
                    self.goal_position.position.y = goal_y_list[self.index]
                elif self.index >= 10:
                    aux_index = random.randrange(0, 11)
                    print(self.index, aux_index)
                    if self.last_index == aux_index:
                        position_check = True
                    else:
                        self.last_index = aux_index
                        position_check = False
                        self.goal_position.position.x = goal_x_list[aux_index]
                        self.goal_position.position.y = goal_y_list[aux_index]

                # self.index = random.randrange(0, 26)
                # #print(self.index, self.last_index)
                # if self.last_index == self.index:
                #     position_check = True
                # else:
                #     self.last_index = self.index
                #     position_check = False

                # self.goal_position.position.x = goal_x_list[self.index]
                # self.goal_position.position.y = goal_y_list[self.index]

        if self.stage == 2:
            while position_check:
                goal_x_list = [0.874355, 0.874355, 0.874355, 0.874355, 0.874355, 0.874355, 0.874355, 0.874355, 0.874355, 0.874355, 0.874355]
                goal_y_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                if not running:
                    self.index = 0
                    position_check = False
                    self.goal_position.position.x = goal_x_list[self.index]
                    self.goal_position.position.y = goal_y_list[self.index]
                elif self.index < 14:
                    self.index += 1
                    position_check = False
                    self.goal_position.position.x = goal_x_list[self.index]
                    self.goal_position.position.y = goal_y_list[self.index]
                elif self.index >= 14:
                    aux_index = random.randrange(0, 15)
                    print(self.index, aux_index)
                    if self.last_index == aux_index:
                        position_check = True
                    else:
                        self.last_index = aux_index
                        position_check = False
                        self.goal_position.position.x = goal_x_list[aux_index]
                        self.goal_position.position.y = goal_y_list[aux_index]


        if self.stage == 3:
            while position_check:
                position_check = False
                self.goal_position.position.x = 0.874355
                self.goal_position.position.y = 0

        time.sleep(0.5)
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y
