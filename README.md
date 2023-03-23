# fourleg_ddpg

This simulation package is meant to train the quadruped robot to walk with the deep deterministic policy gradient (DDPG), 
and also the modified ddpg with the physic informed neural network(PINN).

The objective of this modification is to improve the learning performance for the quadruped walking robot from the DDPG algorithm.
In which PINN has the potential in reducing the estimation error of the neural netowrk through the physical informed neural network.

Other than improvement for the learning algorithm, the locomotion for directional walking behavior training is take into account in this research.
For generalizing the action-space design and speeding the learning process, the information of the walking sequence schematic is provided together with the range of angle value for each joint of the quadruped leg in the training for this research. [This could be refer to the robot_env.py]

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
- Basically all are same as above except main script and the agent script
- robot_ddpg_pinn_main.py: main script for ddpg assisted with pinn
- ddpg_pinn_torch.py : agent in pytorch with the PINN assisted

# Trained model examining in simulation
Noticed that the main script for all the model are the same, but remember to uncomment either one for the agent source import in it 
# DDPG and DDPG with PINN
- learned_robot_main.py: main script for loading trained model
- learned_robot_env.py: robot enviornment
- ddpg_torch.py : agent in pytorch (for DDPG model)
- ddpg_pinn_torch.py : agent in pytorch (for DDPG with PINN model)
- buffer.py : memory buffer
- network.py : network structure 
- noise.py : action noise eliminator
- utils.py : essenstial of plotting code
- tmp: trained model storage

# Trained model examining in experimental work
1. Noticed that the robot should be connected appropiately with either on a single board computer(sbc) such as a raspberry pi for communicate with the remote PC or direct connect to the PC.
2. Noticed that the main script for all the model are the same, but remember to uncomment either one for the agent source import in it 
3. Basically all the script are the same as in the simulation,except the main script and the robot environment script 
4. The robot hardware was setup with dynamixel MX-106 for the actuation and joint state feedback, RPLiDar for scanning the environment, MPU9250/6050 for providing the feedback of robot state.The respective device communication package are listed as in the dependency .

# DDPG and DDPG with PINN
-real_robot_main.py
-real_robot_env.py (for the robot connect with sbc will need to initiate this file at the robot's sbc)

# Dependency package for robot hardware communication
- DynamixelSDK
- dynamixel-workbench
- dynamixel-workbench-msgs
- hls_lfcd_lds_driver
- mpu_6050_driver

# Others dependency for this project
- numpy
- math
- std_msg
- control_msg
- trajectory_msgs
- std_srvs
- geometry_msgs
- sensor_msgs
- gazebo_msgs
- nav_msgs
- nav_msgs
- tf.transformations

# Step for running the training
$ roslaunch fourleg_ddpg hello_gazebo.launch 
$ cd <your_workspace>/fourleg_ddpg/src/
$ ./robot_ddpg_main.py  

# Step for examning the trained model in simulation
$ roslaunch fourleg_ddpg hello_gazebo.launch 
$ cd <your_workspace>/fourleg_ddpg/src/
$ ./learned_robot_main.py

# Step for examning the trained model in expeimental condition
$ roslaunch fourleg_ddpg hello_real_robot.launch 
$ cd <your_workspace>/fourleg_ddpg/src/
$ ./real_robot_main.py



