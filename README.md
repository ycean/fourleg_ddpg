# fourleg_ddpg

This simulation package is meant to train the quadruped robot to walk with the deep deterministic policy gradient (DDPG), 
and also the modified ddpg with the physic informed neural network(PINN).

The objective of this modification is to improve the learning performance for the walking robot from the DDPG algorithm.
In which PINN has the potential in reducing the estimation error of the neural netowrk through the physical informed neural network.

Other than improvement for the learning algorithm, the walking locomotion behavior training is take into account in this research.
For speeding the learning process, the information of the walking sequence schematic is given in the action space training for this research. [This could be refer to the robot_env.py]

Every model trained would be stored into /tmp/ddpg

After model were trained, these model is used to generate for the quadruped walk in both simulation and real robot experiment.
The followoing are the remark for the coding that belong to training, testing in simulation, and examining in real robot .

# Model training
Noticed that in the main script for training, the agent were call to choose action
# DDPG
- robot_ddpg_main.py :main script
- robot_env.py : robot enviornment
- ddpg_torch.py : agent in pytorch
- buffer.py : memory buffer
- network.py : network structure 
- noise.py : action noise eliminator
- respawnGoal_.py : target respawn during training (trained with random target)
- utils.py : essenstial of plotting code

# DDPG with PINN
- robot_ddpg_pinn_main.py: main script for ddpg assisted with pinn
- robot_env.py :
- ddpg_pinn_torch.py : agent in pytorch with the PINN assisted
- buffer.py 
- network.py 
- noise.py 
- respawnGoal_.py 
- utils.py 

# Learned model testing in simulation
Noticed that in the main script for testing, the agent were call to action (which will load the trained model for action)
# DDPG
- learned_robot_main.py: main script for loading trained model
- learned_robot_env.py: robot enviornment
- ddpg_torch.py : agent in pytorch
- buffer.py : memory buffer
- network.py : network structure 
- noise.py : action noise eliminator
- respawnGoal.py : target respawn during simulation (test with special path)
- utils.py : essenstial of plotting code

# DDPG with PINN
- learned_robot_DDPG_PINN_main.py
- learned_robot_env.py
- ddpg_pinn_torch.py : agent in pytorch with the PINN assisted
- buffer.py 
- network.py 
- noise.py 
- respawnGoal.py : target respawn during simulation (test with special path)
- utils.py 

# Learned model testing in real robot
Noticed that in the main script for testing, the agent were call to action (which will load the trained model for action)
