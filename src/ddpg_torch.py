import rospy
import os
import numpy as np
import torch as T
import torch.nn.functional as F
from math import sin, cos, pi, pow, atan2, sqrt, ceil, atan, hypot
from networks import ActorNetwork, CriticNetwork
from noise import OUActionNoise
from buffer import ReplayBuffer


#----- PINN --------#
# PDE as loss function. Thus would use the network which we call as u_theta
#def f(s,a, net):
#    s_sample = s
#    a_sample = a
#    u = net.forward(s_sample,a_sample) # the dependent variable u is given by the network based on independent variables s,a
#    ## Based on our f = du/ds - 2du/da - u, we need du/ds and du/da
#    #u.backward(retain_graph=True, create_graph=True)
#    u_s = T.autograd.grad(u.sum(), s_sample, create_graph=True)[0]
#    #u_s = u_s.mean(1,True)
#    u_s = u_s.abs().sum(1,True)
#    #u_s = s_sample.grad
#    u_a = T.autograd.grad(u.sum(), a_sample, create_graph=True)[0]
#    u_a = u_a.abs().sum(1,True)
#    #u_a = a_sample.grad
#    u_aa = T.autograd.grad(u_a.sum(), a_sample, create_graph=True)[0]
#    u_aa = u_aa.abs().sum(1,True)
#    
#    pde = u_s - u*u_a - (0.01/pi)*u_aa
#    return pde


class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=100, fc1_dims=400, fc2_dims=300, 
                 batch_size=5):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.q_value = []
        self.t_q_value = []
        self.u_loss = []

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='critic')

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_actor')

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_critic')

        #self.net = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
        #                        n_actions=n_actions, name='pinn')
                                
        self.update_network_parameters(tau=1)
                

    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), 
                                    dtype=T.float).to(self.actor.device)
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]
        
    def action_(self, observation):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        action = self.actor.forward(state).to(self.actor.device)
        
        return action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        
    def get_q(self):
    	q_ave = np.average(self.q_value)
    	return q_ave
        
    def get_t_q(self):
        t_q_ave = np.average(self.t_q_value)
        return t_q_ave
        
    def get_uloss(self):
    	uloss_ave = np.average(self.u_loss)
    	return uloss_ave

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        
        if self.memory.mem_cntr < self.batch_size:
            return

        print("learning...")
        states, actions, rewards, states_, done = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float, requires_grad=True).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float, requires_grad=True).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float, requires_grad=True).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float, requires_grad=True).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)
        self.q_value = critic_value.detach().cpu().numpy()

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)
        self.t_q_value = target.detach().cpu().numpy()
        
        #PINN
        #critic_pde = f(states, actions, self.net)
        #pt_all_zeros = critic_pde * 0
        #mse_f = F.mse_loss(critic_pde, pt_all_zeros)
        
        u_loss = F.mse_loss(target, critic_value)
        self.u_loss = u_loss.detach().cpu().numpy()

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value) # + mse_f  
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)





