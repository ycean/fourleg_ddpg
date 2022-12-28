#!/usr/bin/env python
#import gym
#gym.logger.set_level(40)
import rospy
import os
import json
import random
import time
from learned_robot_env import Env
import numpy as np
from ddpg_pinn_torch import Agent
from utils import plot_learning_curve

ACTION_DIMENSION = 14
STATE_DIMENSION = 35  

if __name__ == '__main__':
	rospy.init_node('examine_trained_ddpg_pinn_model')
	print('Node initiate: Test DDPG-PINN models...')
	env = Env(action_dim = ACTION_DIMENSION)
	past_action = np.zeros(ACTION_DIMENSION)
	agent = Agent(alpha=0.0001, beta=0.001, 
                      input_dims=STATE_DIMENSION, tau=0.001,
                      batch_size=1000, fc1_dims=400, fc2_dims=300, 
                      n_actions=ACTION_DIMENSION)
	n_games = 20000
	score_history = []
	q_history = []
	t_q_history = []
	pde_history = []
	uloss_history = []
	pose_history = np.array([])
	goal_history = np.array([])
	action_js_history = np.array([])
	
	score = 0
	q = 0
	t_q = 0
	pde = 0
	goal = False
	
	agent.load_models()
	for i in range(n_games):
		print('Simulating with trained ddpg+pinn...')
		observation = env.reset(goal)
		done = False
		
		while not done:
			action = agent.action_(observation)
			observation_, reward, done, goal = env.step(action,past_action)
			past_action = action
			score += reward
			q_ave = agent.get_q()
			print('q_ave: ', q_ave)
			q_history.append(q_ave)
			target_q_ave = agent.get_t_q()
			print('target_q_ave: ', target_q_ave)
			t_q_history.append(target_q_ave)
			pde_ave = agent.get_pde()
			print('pde_ave: ', pde_ave)
			pde_history.append(pde_ave)
			uloss_ave = agent.get_uloss()
			print('uloss_ave: ', uloss_ave)
			uloss_history.append(uloss_ave)
			#goal_ = env.get_goal_record()
			#goal_history = np.append(goal_history,goal_)
			#goal_history = goal_history.reshape(int(len(goal_history)/2),2)
			#pose_ = env.get_pose_record()
			#pose_history = np.append(pose_history,pose_)
			#pose_history = pose_history.reshape(int(len(pose_history)/3),3)
			#joint_ = env.get_action_js()
			#action_js_history = np.append(action_js_history,joint_)
			#action_js_history = action_js_history.reshape(int(len(action_js_history)/12),12)
			
			
		score_history.append(score)
		filename1 = 'Special_path_test_with_DDPG_PINN3_QuadRob_rewards_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename1, score_history, delimiter=",") # Rewards
		filename2 = 'Special_path_test_with_DDPG_PINN3_q_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename2, q_history, delimiter=",") # q_value
		filename3 = 'Special_path_test_with_DDPG_PINN3_target_q_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename3, t_q_history, delimiter=",") # target_q_value
		filename4 = 'Special_path_test_with_DDPG_PINN3_pde_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename4, pde_history, delimiter=",") # pde_value
		filename5 = 'Special_path_test_with_DDPG_PINN3_u_loss_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename5, uloss_history, delimiter=",") # u_loss_value
		#filename6 = 'Special_path_test_with_DDPG_PINN3_POSE' + '_games_'+ str(agent.batch_size) +'_batch.csv'
		#np.savetxt(filename6, pose_history, delimiter=",") # pose
		#filename7 = 'Special_path_test_with_DDPG_PINN3_GOAL' + '_games_'+ str(agent.batch_size) +'_batch.csv'
		#np.savetxt(filename7, goal_history, delimiter=",") # pose
		#filename8 = 'Special_path_test_with_DDPG_PINN3_action_JS' + '_games_'+ str(agent.batch_size) +'_batch.csv'
		#np.savetxt(filename8, action_js_history, delimiter=",") # action
		
		
	




