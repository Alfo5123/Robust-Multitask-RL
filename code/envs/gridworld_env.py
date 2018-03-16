import gym
import sys
import os
import copy
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from gym.utils import seeding

EMPTY = BLACK = 0
WALL = GRAY = 1
TARGET = GREEN = 3
AGENT = RED = 4
SUCCESS = PINK = 6
COLORS = {BLACK: [0.0, 0.0, 0.0], GRAY: [0.5, 0.5, 0.5], GREEN: [0.0, 1.0, 0.0],
          RED: [1.0, 0.0, 0.0], PINK: [1.0, 0.0, 1.0]}

NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4

class GridworldEnv():
    
    metadata = {'render.modes': ['human','rgb_array']}
    num_env = 0

    def __init__(self, plan):

        self.actions = [NOOP, UP, DOWN, LEFT, RIGHT]
        self.inv_actions = [0, 2, 1, 4, 3]
        self.action_space = spaces.Discrete(5)
        self.action_pos_dict = {NOOP: [0, 0], UP: [-1, 0], DOWN: [1, 0], LEFT: [0, -1], RIGHT: [0, 1]}

        self.img_shape = [256, 256, 3]  # visualize state

        # initialize system state
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.grid_map_path = os.path.join(this_file_path, 'plan{}.txt'.format(plan))
        self.start_grid_map = self._read_grid_map(self.grid_map_path)  # initial grid map
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
        self.grid_map_shape = self.start_grid_map.shape
        self.observation_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0 ]) , \
        									high = np.array([ 1.0, 1.0, 1.0])  )

        # agent state: start, target, current state
        self.agent_start_state, self.agent_target_state = self._get_agent_start_target_state()
        self.agent_state = copy.deepcopy(self.agent_start_state)

        # set other parameters
        self.restart_once_done = False  # restart or not once done
       
        # set seed
        self.seed()

        # consider total episode reward
        self.episode_total_reward = 0.0

        # consider viewer for compatibility with gym
        self.viewer = None

    def seed( self, seed = None):

    	# Fix seed for reproducibility

    	self.np_random, seed = seeding.np_random(seed)
    	return [seed]

    def get_state ( self , coordinates, action, reward ) :

    	# Return a triple with: current location of the agent in the map
    	# given coordinates, the previous action and the previous reward

        ## Normalized for better perform of the NN

    	return np.asarray([ 2.*(self.grid_map_shape[0]*coordinates[0]+coordinates[1]) / ( self.grid_map_shape[0] * self.grid_map_shape[1] ) - 1., \
                             (action-2.5)/5. , reward ] )

    def step(self, action):

        # Return next observation, reward, finished, success

        action = int(action)
        info = {'success': False}
        done = False

        #Penalties
        penalty_step = 0.1
        penalty_wall = 0.5

        reward = -penalty_step 
        nxt_agent_state = (self.agent_state[0] + self.action_pos_dict[action][0],
                           self.agent_state[1] + self.action_pos_dict[action][1])

        if action == NOOP:
            info['success'] = True
            self.episode_total_reward += reward #Update total reward
            return self.get_state(self.agent_state, action, reward), reward, False, info

	#Make a step
        next_state_out_of_map = (nxt_agent_state[0] < 0 or nxt_agent_state[0] >= self.grid_map_shape[0]) or \
                                (nxt_agent_state[1] < 0 or nxt_agent_state[1] >= self.grid_map_shape[1])

        if next_state_out_of_map:
            info['success'] = False
            self.episode_total_reward += reward #Update total reward
            return self.get_state(self.agent_state, action, reward), reward, False, info

        # successful behavior
        target_position = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]

        if target_position == EMPTY:

            self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = AGENT

        elif target_position == WALL:

            info['success'] = False
            self.episode_total_reward += (reward-penalty_wall) #Update total reward
            return self.get_state(self.agent_state, action, reward-penalty_wall), (reward-penalty_wall), False, info

        elif target_position == TARGET:

            self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = SUCCESS

        self.current_grid_map[self.agent_state[0], self.agent_state[1]] = EMPTY
        self.agent_state = copy.deepcopy(nxt_agent_state)
        info['success'] = True

        if nxt_agent_state[0] == self.agent_target_state[0] and nxt_agent_state[1] == self.agent_target_state[1]:
            done = True
            reward += 1.0
            if self.restart_once_done:
                self.reset()

        self.episode_total_reward += reward #Update total reward
        return self.get_state(self.agent_state, action, reward), reward, done, info

    def reset(self):

    	# Return the initial state of the environment

        self.agent_state = copy.deepcopy(self.agent_start_state)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.episode_total_reward = 0.0
        return self.get_state(self.agent_state, 0.0, 0.0)

    def close(self):
        if self.viewer: self.viewer.close()

    def _read_grid_map(self, grid_map_path):

    	# Return the gridmap imported from a txt plan

        grid_map = open(grid_map_path, 'r').readlines()
        grid_map_array = []
        for k1 in grid_map:
            k1s = k1.split(' ')
            tmp_arr = []
            for k2 in k1s:
                try:
                    tmp_arr.append(int(k2))
                except:
                    pass
            grid_map_array.append(tmp_arr)
        grid_map_array = np.array(grid_map_array, dtype=int)
        return grid_map_array

    def _get_agent_start_target_state(self):
        start_state = np.where(self.start_grid_map == AGENT)
        target_state = np.where(self.start_grid_map == TARGET)

        start_or_target_not_found = not(start_state[0] and target_state[0])
        if start_or_target_not_found:
            sys.exit('Start or target state not specified')
        start_state = (start_state[0][0], start_state[1][0])
        target_state = (target_state[0][0], target_state[1][0])

        return start_state, target_state

    def _gridmap_to_image(self, img_shape=None):

    	# Return image from the gridmap

        if img_shape is None:
            img_shape = self.img_shape
        observation = np.random.randn(*img_shape) * 0.0
        gs0 = int(observation.shape[0] / self.current_grid_map.shape[0])
        gs1 = int(observation.shape[1] / self.current_grid_map.shape[1])
        for i in range(self.current_grid_map.shape[0]):
            for j in range(self.current_grid_map.shape[1]):
                for k in range(3):
                    this_value = COLORS[self.current_grid_map[i, j]][k]
                    observation[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1, k] = this_value
        return (255*observation).astype(np.uint8)

    def render(self, mode='human',close = False):

    	# Returns a visualization of the environment according to specification

        if close: 
            plt.close(1) #Final plot
            return

        img = self._gridmap_to_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
        	plt.figure()
        	plt.imshow(img)
        	return
