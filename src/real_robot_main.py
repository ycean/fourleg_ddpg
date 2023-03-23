#!/usr/bin/env python
#import gym
#gym.logger.set_level(40)
import rospy
import os
import json
import random
import time
from real_robot_env import Env
import numpy as np
from ddpg_pinn_torch import Agent
#from ddpg_torch import Agent
from utils import plot_learning_curve
#import socket
import sys

## Create a TCP/IP socket
#sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

ACTION_DIMENSION = 14
STATE_DIMENSION = 35  

if __name__ == '__main__':
	rospy.init_node('real_robot_walk_with_ddpg_pinn_b_model')
	print('Node initiate: Test real robot walk with ddpg_pinn_b model...')
	env = Env(action_dim = ACTION_DIMENSION)
	past_action = np.zeros(ACTION_DIMENSION)
	agent = Agent(alpha=0.0001, beta=0.001, 
                      input_dims=STATE_DIMENSION, tau=0.001,
                      batch_size=1, fc1_dims=400, fc2_dims=300, 
                      n_actions=ACTION_DIMENSION)
	n_games = 300
	score_history = []
	q_history = []
	t_q_history = []
	uloss_history = []
	pose_history = np.array([])
	goal_history = np.array([])
	action_js_history = np.array([])
	score = 0
	q = 0
	t_q = 0
	goal = False
	
	agent.load_models()
	for i in range(n_games):
		print('Simulating with trained ddpg at game ', i)
		observation = env.reset()
		done = False
		while not done:
			action = agent.action_(observation)
			observation_, reward, done= env.step(action,past_action)
			past_action = action
			score += reward
			q_ave = agent.get_q()
			print('q_ave: ', q_ave)
			q_history.append(q_ave)
			target_q_ave = agent.get_t_q()
			print('target_q_ave: ', target_q_ave)
			t_q_history.append(target_q_ave)
			uloss_ave = agent.get_uloss()
			print('uloss_ave: ', uloss_ave)
			uloss_history.append(uloss_ave)
			
			
			
		score_history.append(score)
		avg_score = np.mean(score_history[-100:])
		print('episode ', i, 'score %.1f' % score,'average score %.1f' % avg_score)
		filename1 = 'Real_robot_test_with_ddpg_pinn_b_rewards_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename1, score_history, delimiter=",") # Rewards
		filename2 = 'Real_robot_test_with_ddpg_pinn_b_q_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename2, q_history, delimiter=",") # q_value
		filename3 = 'Real_robot_test_with_ddpg_pinn_b_target_q_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename3, t_q_history, delimiter=",") # target_q_value
		filename4 = 'Real_robot_test_with_ddpg_pinn_b_u_loss_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename4, uloss_history, delimiter=",") # u_loss_value

		
		
	




