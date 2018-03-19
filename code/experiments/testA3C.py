import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv
sys.path.append('../a3c')
import trainingA3C

import gym 

trainingA3C.trainA3C(file_name="env3", 
					 env= GridworldEnv(3),
					 update_global_iter=10,
            		 gamma=0.95, 
            		 is_plot=False, 
            		 num_episodes=500, 
            		 max_num_steps_per_episode=1000, 
            		 learning_rate=0.001 )
