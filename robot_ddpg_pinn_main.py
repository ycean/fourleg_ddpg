#!/usr/bin/env python
#import gym
#gym.logger.set_level(40)
import rospy
import os
import json
import random
import time
from robot_env import Env
import numpy as np
from ddpg_pinn_torch import Agent
from utils import plot_learning_curve

ACTION_DIMENSION = 14
STATE_DIMENSION = 35  

if __name__ == '__main__':
	rospy.init_node('ddpg_QuadRob')
	print('Node initiate: ddpg_QuadRob..')
	env = Env(action_dim = ACTION_DIMENSION)
	past_action = np.zeros(ACTION_DIMENSION)
	agent = Agent(alpha=0.0001, beta=0.001, 
                      input_dims=STATE_DIMENSION, tau=0.001,
                      batch_size=1000, fc1_dims=400, fc2_dims=300, 
                      n_actions=ACTION_DIMENSION)
	n_games = 1000
	batch_size=1000
	filename = 'Quadrob_modified_ddpg' + str(agent.alpha) + '_beta_' + str(agent.beta) + '_' + str(n_games) + '_games_' + str(agent.batch_size) +'_batch'
	figure_file = 'plots/' + filename + '.png'
	score_history = []
	q_history = []
	t_q_history = []
	pde_history = []
	score = 0
	q = 0
	t_q = 0
	for i in range(n_games):
		print('Training...')
		observation = env.reset()
		done = False
		q_ave = 0
		target_q_ave = 0
		
		agent.noise.reset()
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done = env.step(action,past_action)
			past_action = action
			agent.remember(observation, action, reward, observation_, done)
			agent.learn()
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
			observation = observation_
			
		score_history.append(score)
		avg_score = np.mean(score_history[-100:])
		agent.save_models()
		print('Models saving...')
		print('episode ', i, 'score %.1f' % score,'average score %.1f' % avg_score)
		filename1 = 'QuadRob_modified_ddpg_rewards_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename1, score_history, delimiter=",") # Rewards
		filename2 = 'QuadRob_modified_ddpg_q_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename2, q_history, delimiter=",") # q_value
		filename3 = 'QuadRob_modified_ddpg_target_q_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename3, t_q_history, delimiter=",") # target_q_value
		filename3 = 'QuadRob_modified_ddpg_pde_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename3, pde_history, delimiter=",") # pde_value
			
	x = [i+1 for i in range(n_games)]
	plot_learning_curve(x, score_history, figure_file)
	#filename1 = 'QuadRob_ddpg_rewards_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
	#np.savetxt(filename1, score_history, delimiter=",") # Rewards




