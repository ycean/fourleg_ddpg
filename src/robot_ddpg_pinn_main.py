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
	n_games = 20000
	batch_size=1000
	filename = 'QuadRob_ddpg_pinn_b' + str(agent.alpha) + '_beta_' + str(agent.beta) + '_' + str(n_games) + '_games_' + str(agent.batch_size) +'_batch'
	figure_file = 'plots/' + filename + '.png'
	score_history = []
	q_history = []
	t_q_history = []
	pde_history = []
	uloss_history = []
	reward_history = []
	total_reward_history = []
	steps_history = []
	reward = 0
	score = 0
	q = 0
	t_q = 0
	for i in range(n_games):
		print('Training...')
		observation = env.reset()
		done = False
		q_ave = 0
		target_q_ave = 0
		steps = 0
		total_reward = 0
		
		agent.noise.reset()
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done = env.step(action,past_action)
			past_action = action
			agent.remember(observation, action, reward, observation_, done)
			agent.learn()
			score += reward
			total_reward += reward
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
			observation = observation_
			steps += 1
			
		score_history.append(score)
		reward_history.append(reward)
		steps_history.append(steps)
		total_reward_history.append(total_reward)
		avg_score = np.mean(score_history[-100:])
		agent.save_models()
		print('Models saving...')
		print('episode ', i, 'reward %.1f' % reward, 'step numbers %.1f' % steps)
		print('episode ', i, 'score %.1f' % score,'average score %.1f' % avg_score)
		filename1 = 'QuadRob_ddpg_pinn_b_culmulative_rewards_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename1, score_history, delimiter=",") # Culmulative_Rewards
		filename2 = 'QuadRob_ddpg_pinn_b_q_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename2, q_history, delimiter=",") # q_value
		filename3 = 'QuadRob_ddpg_pinn_b_target_q_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename3, t_q_history, delimiter=",") # target_q_value
		filename4 = 'QuadRob_ddpg_pinn_b_pde_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename4, pde_history, delimiter=",") # pde_value
		filename5 = 'QuadRob_ddpg_pinn_b_u_loss_value_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename5, uloss_history, delimiter=",") # u_loss_value
		filename6 = 'QuadRob_ddpg_pinn_b_REWARD_in_end_ep' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename6, reward_history, delimiter=",") # REWARD
		filename7 = 'QuadRob_ddpg_pinn_b_Step_number_record' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename7, steps_history, delimiter=",") # step_number_in_single_episode
		filename8 = 'QuadRob_ddpg_pinn_b_Total_REWARD_in_ep' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
		np.savetxt(filename8, total_reward_history, delimiter=",") # total REWARD in single episode
			
	x = [i+1 for i in range(n_games)]
	plot_learning_curve(x, score_history, figure_file)
	#filename1 = 'QuadRob_ddpg_rewards_' + str(n_games) + '_games_'+ str(agent.batch_size) +'_batch.csv'
	#np.savetxt(filename1, score_history, delimiter=",") # Rewards




