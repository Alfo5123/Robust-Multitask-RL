import sys
sys.path.append('../')
from envs.gridworld_env import GridworldEnv
from utils import play_game
sys.path.append('../dqn0')
import trainingDQN0

import gym
agent, _, _ = trainingDQN0.trainDQN0(file_name="env8",
                    env=GridworldEnv(8),
                    batch_size=128,
                    gamma=0.9,
                    eps_start=0.9,
                    eps_end=0.05,
                    eps_decay=10000,
                    is_plot=False,
                    num_episodes=500,
                    max_num_steps_per_episode=10000,
                    learning_rate=0.001,
                    memory_replay_size=10000,
                )

#play_game(GridworldEnv(1), agent)
