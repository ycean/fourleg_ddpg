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
from ddpg_torch import Agent
from utils import plot_learning_curve

ACTION_DIMENSION = 14
STATE_DIMENSION = 35  

if __name__ == '__main__':
	rospy.init_node('examine_trained_ddpg_model')
	print('Node initiate: Test DDPG models...')
	env = Env(action_dim = ACTION_DIMENSION)
	past_action = np.zeros(ACTION_DIMENSION)
	agent = Agent(alpha=0.0001, beta=0.001, 
                      input_dims=STATE_DIMENSION, tau=0.001,
                      batch_size=25, fc1_dims=400, fc2_dims=300, 
                      n_actions=ACTION_DIMENSION)
	n_games = 20000
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
		observation = env.reset(goal)
		done = False
		while not done:
			action = agent.action_(observation)
			observation_, reward, done, goal = env.step(action,past_action)
			goal = goal
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
		avg_score = np.mean(score_history[-100:])
		print('episode ', i, 'score %.1f' % score,'average score %.1f' % avg_score)
		filename1 = 'Special_path_test_ddpg_rewards_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename1, score_history, delimiter=",") # Rewards
		filename2 = 'Special_path_test_ddpg_q_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename2, q_history, delimiter=",") # q_value
		filename3 = 'Special_path_test_ddpg_target_q_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename3, t_q_history, delimiter=",") # target_q_value
		filename4 = 'Special_path_test_ddpg_u_loss_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename4, uloss_history, delimiter=",") # u_loss_value
		#filename6 = 'Special_path_test_with_DDPG_POSE_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		#np.savetxt(filename6, pose_history, delimiter=",") # pose
		#filename7 = 'Special_path_test_with_DDPG_GOAL' + '_games_'+ str(agent.batch_size) +'_batch.csv'
		#np.savetxt(filename7, goal_history, delimiter=",") # pose
		#filename8 = 'Special_path_test_with_DDPG_action_JS_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		#np.savetxt(filename8, action_js_history, delimiter=",") # action
		
		
	




